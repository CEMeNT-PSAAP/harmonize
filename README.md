# Harmonize

Harmonize is an execution framework for GPU with the aims of increasing performance and decreasing development burden for applications requiring complex processing patterns.
Harmonize is currently available both as a headers-only CUDA C++ library and as a Python package. In both forms, functionality is exposed as an asynchronous processing framework.

## Fundamentals

The system of asynchronous functions that are executed within a given runtime are defined in advance through a **program specification**.
A program specification only represents these systems abstractly, but they may be transformed into an implementation by declaring a **program** ***specialization***.

Program specifications are defined by application developers, and represent the business logic of what needs to be accomplished in the GPU, whereas the templates (**program types**) used to transform them into specializations are defined by framework developers.
This structure is used because it separates much of the underlying implementation details away from application developers, delegating those concerns to the framework developers.
Ideally, with the addition of a program type, transitioning a codebase to this other program could be as little as a one-line refactor:

```
< using MyProgram = EventProgram<MyProgramSpec>;
> using MyProgram = AsyncProgram<MyProgramSpec>;
```

Likewise, as long as the interface between the specification and the program type does not change, program types may be updated without requiring refactors from the application developers.

## Why Async?

Asynchronous programming represents a looser contract between program and execution environment.
Once a function is called, there is no guarantee of where or when that call is actually evaluated.
If the call is lazy, then the evaluation may not happen at all.

This looser contract is useful, because it allows an interface to represent a wide variety of execution strategies without breaking any promises.
By representing all program types as different asynchronous runtimes, they can be used interchangeably as long as they fulfill the few promises made by the runtime's interface.


## Harmonize on AMD?

An AMD-compatible version of Harmonize is in development, but is currently not available as a full or experimental release.


## Dependencies

### CUDA C++ Dependencies

The CUDA C++ framework currently requires:

- a CUDA compiler
- the CUDA runtime (host and device)



### Python Dependencies

Python bindings require:

- Non-package Dependencies
  - `nvcc`
  - the CUDA runtime (host and device)

- Python Package Dependencies
  - `numpy`
  - `numba`
  - `llvmlite`



