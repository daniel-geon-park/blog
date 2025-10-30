---
layout: post
title: '<a href="https://github.com/daniel-geon-park/triton_bwd">`triton_bwd`</a>: Enabling Backpropagation for the OpenAI Triton language'
---

Everyone knows the [OpenAI Triton language](https://triton-lang.org/) is amazing. Writing CUDA kernels in CUDA C++ is such a pain, so writing Python in a pytorch-like syntax that compiles down to GPU machine code and getting the same blazingly-fast performance is such a godsend.

But one problem is that unlike PyTorch, automatic differentiation (AD) is not supported in Triton. This is understandable, but when I'm trying a new ML algorithm with a custom operation, such as [hip-attn](https://github.com/DeepAuto-AI/hip-attention), I also want it to be differentiable so that I can train the model.

In the end, you have to hand-write the backward kernel as well for the best performance, but verifying that my hand-rolled backward kernel actually computes the mathematical derivative of my hand-rolled forward kernel is no easy task. Even if you get it right, each time you change the forward algorithm even a bit, you have to go through the whole debugging process again. But if I could get the correct gradients for my kernel, that would be a great improvement to the situation because then I can at least *check* if my backward kernel is correct or not. Otherwise, I am just running in the dark, with my fingers crossed, and hoping my backward kernel is correct and my model doesn't explode at step 5000 in the training process.

So I created a small library, [`triton_bwd`](https://github.com/daniel-geon-park/triton_bwd), which takes an existing kernel written in Triton, and makes the kernel differentiable in the regular Pytorch way. All you have to do is change the decorator and specify which arguments are in args and which are out args:
```diff
+from triton_bwd import triton_bwd, autotune

-@triton.autotune(...)
-@triton.jit
+@autotune(...)
+@triton_bwd(in_args=["a", "b"], out_args=["c"])
def my_triton_kernel(a, stride_a, b, stride_b, c, stride_c):
    ...
```

```diff
def compute_something(a, b):
    c = torch.zeros_like(a)
-   my_triton_kernel[grid](
+   c, = my_triton_kernel.forward(
+       grid,
        a, a.stride(0),
        b, b.stride(0),
        c, c.stride(0),
    )

    return c  # is now differentiable!
```

...and it works like magic. Of course, the backward pass is not automatically optimized, which makes it horrendously slow for larger sizes of input and quickly blows up the GPU's memory, so you will still have to write the backward kernel yourself. <strong>This library is intended for checking the correctness of your hand-rolled backward kernel by comparing its output against the 'ground truth' derivative values computed by using this library.</strong> Also, this is more of a proof-of-concept, and a lot of functions are missing from the implementation. However, I am using this to check my own backward kernels in `hip-attn`, and I can tell you that it works.

It is worth mentioning that there is another similar AD library for Triton, called [`triton-autodiff`](https://github.com/srush/triton-autodiff). The catch is, it doesn't support control flow, reductions, `tl.load`/`store`, etc. Instead, it only operates on a pure function written in Triton that takes `tl.tensor`s as input and returns a `tl.tensor` as output. This severely limits the range of kernels that this library can operate on. In contrast, my approach works with `tl.load`/`store`, which means my library can be applied to whole kernels rather than small parts of it.

So, how does `triton_bwd` work?

## The Basic Idea

Pytorch already offers a full-blown AD system. So, if we somehow convert a Triton kernel into a sequence of Pytorch operations, we can easily compute the backward pass of it.

Conceptually, a Triton kernel function is called in parallel by multiple threads, where each thread only differs in the return values of `tl.program_id()`. The first thread gets `tl.program_id(0) == 0` and the second gets `tl.program_id(0) == 1` and so on, and each thread has to figure out which parts of the input and output it needs to operate on based on the program id. In other words, based on the program id, the kernel calls `tl.load()` with different parameters to load a specific part of the memory, does a bunch of operations on it, and calls `tl.store()` to update a specific part of the memory with the result. Conceptually, `tl.load()` is very similar to what `torch.gather()` does, and `tl.store()` is very similar to what `torch.scatter()` does.

If we look at a single thread invocation of a Triton kernel, it can be seen as a pure function that takes input tensors, parameters, and crucially, the program id, and returns a list of *updates*, where each update is a tuple of `(pointers, values)`, telling the memory to update the contents of the memory pointed by `pointers` to `values`. Since `tl.store()` can be called multiple times in a single invocation, the output is a *list* of these updates rather than a single update.

In other words, conceptually, we can rewrite a single thread invocation of a Triton kernel into a pure Pytorch function, like so:
```python
def single_invocation(
        # list of tensors that are used by tl.load()
        in_tensors: list[torch.Tensor],
        # scalar arguments
        other_params: dict,
        # the program id
        program_id: Tuple[int, int, int],
) -> list[Tuple[torch.LongTensor, torch.Tensor]]:
    
    # Compute which indices to load from
    indices_to_load = compute_load_indices(program_id, other_params)

    # Instead of tl.load()
    tensor = torch.gather(in_tensors[0], indices_to_load)
    
    output = do_a_bunch_of_computations(tensor, other_params)

    # Compute which indices to store to
    indices_to_store = compute_store_indices(program_id, other_params)
    
    return [(indices_to_store, output)]  # instead of tl.store()
```

Actually converting a triton function into this format can be done by traversing through the triton function's AST and calling Pytorch functions whenever you encounter an operation between triton tensors.

Now, since we have a single invocation in a pytorch format, all we have to do to get the output of all threads is to call this function for every thread, and apply the updates at once:
```python
def simulated_kernel_call(grid, in_tensors, out_tensors, other_params):
    all_updates = []
    # Loop over the grid
    for pid0 in range(grid[0]):
        for pid1 in range(grid[1]):
            for pid2 in range(grid[2]):
                program_id = (pid0, pid1, pid2)
                updates = single_invocation(
                    in_tensors, other_params, program_id
                )
                all_updates.append(updates)  # store 
    # Applies all updates using torch.scatter()
    return apply_updates(out_tensors, all_updates)
```
For efficiency's sake, instead of the nested for loops, we can use [`torch.func.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.func.vmap.html) to do the multiple invocations in parallel.

Now, do you see what I did there? I have created a pure Pytorch function that simulates an entire Triton kernel call. But unlike a Triton kernel call, since it's just a regular Pytorch function, you can call `out_tensors.backward()` or `torch.func.grad()` on it to get the gradient by automatic differentiation.

## Dealing with Dynamic For Loops

One problem arises, however, when dynamic for loops are involved. I get around this by requiring a `max_iters` argument which must be supplied with a static value in a for loop, like so:
```python
for i in range(0, some_dynamic_value, max_iters=SOME_STATIC_VALUE):
    ...
```

I have an idea on how to relax this requirement in the future, but that is for another time.

So that's the basic explanation of how `triton_bwd` works. If you have any ideas on how to improve this library, please add an issue on [the repository](https://github.com/daniel-geon-park/triton_bwd). Any contributions are welcome!
