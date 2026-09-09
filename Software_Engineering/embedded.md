---
title: "Embedded Systems Overview"
date: "2026"
author: "Ke Zhang"
---

# Embedded Systems Overview

[toc]

## Recap of Operating Systems

An **operating system (OS)** is software that manages computer hardware and software resources and provides common services for computer programs, e.g., memory management, process scheduling, and input/output operations. OS sits between the hardware and application software, acting as an intermediary to facilitate efficient resource utilization and program execution.

Remarks:

* Typical hardware resources: CPU, memory, I/O devices
* Typical software resources: device drivers, libraries

### Process and Thread

A **process** is an independent execution instance with its own protected memory space and system resources.

Remarks:

* *memory space*: each process has its own virtual address space containing code (text), globals, heap, and stacks.
* *protected*: the memory space of a process is isolated from other processes by the hardware **memory management unit (MMU)**.
* *system resources*: each process has its own set of system resources, such as file descriptors, network sockets.

The OS manages each process by maintaining metadata such as its process ID (**PID**), parent process ID (**PPID**), process state, scheduling information, and resource usage statistics. Those metadata is stored in the process control block (**PCB**).

Example: On Mac, you can check the process metadata in activity monitor or by using the `ps` command in the terminal.

A **thread** is the smallest execution unit within a process, sharing the parent process's memory space and system resources while having its own execution context.

Remarks:

* *execution context*: each thread has its own program counter, stack, and registers.
* *shared memory space*: threads within the same process can directly access the process's memory, facilitating inter-thread communication but requiring synchronization mechanisms to avoid race conditions.

Example: Suppose you wrote and compiled a hello world program. Running it will create a new single-threaded process. Running it twice at the same time will create two independent processes, each with its own memory space and system resources.

**Inter-process communication (IPC)** allows processes to exchange data, despite having isolated memory spaces. Common IPC mechanisms include pipes, message queues, shared memory, and sockets.

### Memory Management

**Virtual memory** is a memory management technique that gives a process the illusion of having a large, contiguous block of memory, even if the physical memory is fragmented or smaller than the virtual address space. Briefly, this effect is achieved through:

* paging: the virtual address space is divided into fixed-size pages, which are mapped to physical memory frames by the OS and MMU. (Page and frame are just a continuous, equal-sized block of memory in different address spaces.)
* swapping: when physical memory is full, some pages are temporarily moved to disk storage to free up space for other pages.

Illustration of process and thread memory layout (virtual address space):

```txt
      Process A                   Process B                   Process C
Virtual address space       Virtual address space       Virtual address space

high addresses              high addresses              high addresses
┌────────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ Thread A.1 stack   │      │ Thread B.1 stack   │      │ Thread C.1 stack   │
│        ↓           │      │        ↓           │      │ Thread C.2 stack   │
│ Thread A.2 stack   │      │                    │      │        ↓           │
│        ↓           │      │                    │      │                    │
│ Thread A.3 stack   │      │                    │      │                    │
│        ↓           │      │                    │      │                    │
│                    │      │                    │      │                    │
│        ↑           │      │        ↑           │      │        ↑           │
│       heap         │      │       heap         │      │       heap         │
│                    │      │                    │      │                    │
│ globals / data     │      │ globals / data     │      │ globals / data     │
│ code / text        │      │ code / text        │      │ code / text        │
└────────────────────┘      └────────────────────┘      └────────────────────┘
low addresses               low addresses               low addresses
```

## Mainstream Embedded Systems

Embedded systems can be briefly categorized into three major types:

1. **Bare-metal Embedded Systems**: The software runs directly on the hardware without an operating system. The application is typically a main loop driven by interrupts.
    * example: Most Arduino boards and STM32 microcontrollers (MCU) without OS
2. **Real-time Operating System (RTOS) Based Embedded Systems**: The software runs on top of the RTOS kernel, which manages hardware resources and task scheduling.
    * example: STM32 MCU running FreeRTOS or Azure RTOS (ThreadX)
3. **Embedded Linux Systems**: The software runs on top of an embedded Linux kernel, which resembles a full-fledged operating system.
    * example: Raspberry Pi running Ubuntu

Scheduling across different types of embedded systems:

| Scheduling | Bare-metal | RTOS-Based | Embedded Linux |
|------------|------------|-------------|-----------------|
| **Execution model** | main loop + ISR* | managed by RTOS kernel | managed by Linux kernel |
| **Scheduler** | ISR-driven | RTOS scheduler | Linux scheduler |

Memory management across different types of embedded systems:

| Memory Management | Bare-metal | RTOS-Based | Embedded Linux |
|-------------------|------------|-------------|-----------------|
| **Virtual memory** | not supported | usually not supported | fully supported |
| **Address space** | usually one | usually one | separate for each process |
| **Memory protection** | usually none | usually none or MPU* | yes, with MMU |

### Cross-compiling

**Cross-compiling** is the process of compiling code on the host machine (e.g., a laptop or desktop) for execution on a different target machine (e.g., an embedded device).

Remarks:

* The host machine and target machine may have different hardware architectures and OS. e.g. compiling on an x86_64 Windows laptop for an ARM-based STM32 MCU running RTOS.
* Cross-compiling typically requires the host machine to have a **cross-compiler** (or toolchain) that can generate binaries for the target machine.

| Cross-compiler | Host machine | Target machine |
|----------------|------------|----------------|
| arm-none-eabi-gcc | x86_64 (Linux/macOS/Windows) | ARM Cortex-M (bare-metal/RTOS) |
| aarch64-linux-gnu-gcc | x86_64 (Linux) | ARM64 (Linux) |

## Bare-metal Embedded Systems

## RTOS-Based Embedded Systems

## Embedded Linux Systems
