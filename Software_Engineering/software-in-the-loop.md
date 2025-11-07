---
title: "SIL Test"
date: "2025"
author: "Kezhang"
---

Proof-read my notes about SIL test, which is for me a quite new topic. There might be misunderstanding of the subject. Read it carefully. Address the inline questions.

# Software in the Loop Test

Consider an embedded software running on a microcontroller (uC) that connects to multiple peripherals. In unit tests, we test the software units in virtual environments such as virtual machine or docker container.

In integration test, we would like to test the overall system or a subsystem. How to test the the system without the physical hardware? We need some virtual environment that does not only simulates the target uC, but also the topology between the uC and its peripherals. This is exactly the basic idea of **software-in-the-loop (SIL) testing**.

CANoe is a common tool for SIL testing. We can build a simulation environment in CANoe that mimics the structure and interfaces of the real system.

The **system under test (SUT)** is the compiled software to be tested. The SUT is also referred to as software image or binary.

A **SIL adapter** is another piece of software that connects the SUT to the simulation environment. It serves as the bridge between the SUT and CANoe (e.g., for I/O, network frames, timing, and stimuli). In particular, it is responsible for data exchange between the SUT and CANoe.

Basic workflow of SIL test:

1. Preparation:
    * install CANoe (with SIL adapter support)
    * install CMake or Visual Studio (for building the SIL adapter)
1. Build the SUT for SIL
    * replace the hardware dependent modules with simulated counterparts.
    * build the software binaries (aka SUT) for SIL test
    * ensure the software is runnable in uC emulator
1. Build SIL adapter
    * define the interfaces in CANoe
    * map the interfaces to software functions
    * build the adapter using CMake or Visual Studio
1. Connect the SIL adapter to the software
1. Create CANoe simulation environment
    * create network simulation (CAN, LIN, Ethernet, etc.)
    * add peripherals (sensors, actors)
1. Connect the SIL adapter to the simulation environment
1. Define test cases in CAPL
    * write stimuli, checks, and verdicts (PASS/FAIL)
1. Run and debug the system
    * execute measurements, inspect traces, log failures, and iterate
    * optional: automate testing

## Step-by-Step Guide

Example: embedded software running on an arm processor, which connects to a rotary switch and an LED. As the state of rotary switch is changed, the processor sets a different blink pattern to the LED.

### Prepare the Source Code for SIL

The production code typically calls **hardware abstraction layer (HAL)** functions for accessing the uC regiesters and peripherals. Before SIL test, we must replace those HAL functions with **stub functions** (or simply **stubs**) which simulate the behaviour of the hardware. As the result, the SIP build of the SUT is runable on the host PC without physical hardware.

Remarks:

* The stubs are also known as **adapter functions** or **mock HAL**. They are simulation-based implementations of the HAL APIs.
* In SIL, the SUT must be built and linked against these replacements. Otherwise, the SUT will fail due to missing hardware access. The SUT must be a standalone software runnable on the host PC.
* 💣 Not to be confused by the wording *simulation*:
  * Stubs simulates the behaviour of individual hardware
  * CANoe simulates the environment which the SUT interacts with
* 💣 Stubs are not tied to CANoe. They are indepedently runnable on the host PC.

The best practice for writing stubs:

1. keep the HAL API intact for portability
1. provide a simulated implementation of the HAL functions. This can be done either via conditional compiliation (e.g. `#ifdef SIL_BUILD`) or separate source files.

With all HAL functions being replaced, we can build the binary aka the SUT. The format of SUT depends on the executation strategy:

* host-native build: build the SUT as a Windows/Linux executable or library. On windows, SUT is typically built as `.exe` or `.dll`.
* instruction-set simulation (ISS): too advanced. not covered here.

Question:

1. show a simple example of how a HAL function is replaced with a corresponding stub.

### Create SIL Adapter

On windoes, SIL adpter is essentially a dynamically linked library (`.dll` file).
To create a SIL adapter, we new a DLL project, in which we include

* CANoe SIL headers (provided by Vector)
* Our SUT-specific headers.

Then, we need to implement the adapter interface functions:

* `Init()` – Initialize SUT and adapter.
* `Start()` – Start execution.
* `Stop()` – Stop execution.
* `Process()` – Called cyclically by CANoe to exchange data.
* `Shutdown()` – Clean up resources.

Questions:

1. The adapter interface functions like `Init()` are built-in API from Vector, right? Should we always implement them? Are above five adapter interface functions the only ones we need to implement?
1. Should we build the adapter? Previously, we built the SUT. These are distinct two things, right?