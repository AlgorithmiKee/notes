---
title: "SIL Test"
date: "2025"
author: "Kezhang"
---

Proof-read my notes about SIL test, which is for me a quite new topic. There might be misunderstanding of the subject. Read it carefully and make sure everthing is written precisely and rigorously. Address the inline questions.

# Software in the Loop Test

Consider an embedded software running on a microcontroller (uC) that connects to multiple peripherals. In unit tests, we test the software units in virtual environments such as virtual machine or docker container.

In integration test, we would like to test the overall system or a subsystem. How to test the the system without the physical hardware? We need some virtual environment that does not only simulates the target uC, but also the topology and interaction between the uC and its peripherals. This is exactly the basic idea of **software-in-the-loop (SIL) testing**.

**CANoe** is a common tool for SIL testing. It allows building a simulation environment that mimics the structure and interfaces of the real system.

## The Big Picture

In SIL, the **system under test (SUT)** is a custom build of the software to be tested. The SUT is also referred to as software image or binary. Inside SUT:

* hardware-dependent code is replace with **stubs**, which simulates the hardware-level function calls (e.g. GPIO reads).

* SUT contains the main test **logic** (e.g. how the uC reacts to certain sensor inputs).

A **SIL adapter** is another piece of software that connects the SUT to the simulation environment. It serves as the bridge between the SUT and CANoe (e.g., for I/O, network frames, timing, and stimuli). In particular, it is responsible for data exchange between the SUT and CANoe.

During test execution:

1. CANoe loads the SIL adapter.
1. SIL adapter loads and runs the SUT.
1. CANoe sends test stimuli into SIL adpater, which converts the test stimuli into function calls or data understood by the SUT.
1. The SUT receives the input from the SIL adapter, processes it, and reacts to it by sending outputs.
1. The SIL adapter converts the SUT outputs back to CANoe for monitoring and evaluation.

## Step-by-Step Guide

Example: embedded software running on an arm processor, which connects to a rotary switch and an LED. As the state of rotary switch is changed, the processor sets a different blink pattern to the LED.

Basic workflow of SIL test:

1. Preparation:
    * install CANoe (with SIL adapter support)
    * install CMake or Visual Studio (for building the SIL adapter)
1. Build the SUT for SIL
    * replace the hardware dependent modules with stubs.
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

### Prepare the Source Code for SIL

The production code typically calls **hardware abstraction layer (HAL)** functions for accessing the uC regiesters and peripherals. Before SIL test, we must replace those HAL functions with **stub functions** (or simply **stubs**) which simulate the behaviour of the hardware. As the result, the SIL build of the SUT is runable on the host PC without physical hardware.

Remarks:

* The stubs are also known as **adapter functions** or **mock HAL**. They are simulation-based implementations of the HAL APIs.
* In SIL, the SUT must be built and linked against these replacements. Otherwise, the SUT will fail due to missing hardware access. The SUT must be a standalone software runnable on the host PC.
* 💣 Not to be confused by the wording *simulation*:
  * Stubs simulate the internal hardware-dependent behaviour of SUT
  * CANoe simulates the external environment which the SUT interacts with
* 💣 Stubs are not tied to CANoe. They are indepedently runnable on the host PC.

The best practice for writing stubs:

1. keep the HAL API intact for portability
1. provide a simulated implementation of the HAL functions. This can be done either via conditional compiliation (e.g. `#ifdef SIL_BUILD`) or separate source files.

With all HAL functions being replaced, we can build the binary aka the SUT. The format of SUT depends on the executation strategy:

* host-native build: build the SUT as a Windows/Linux executable or library. On windows, SUT is typically built as `.exe` or `.dll`.
* instruction-set simulation (ISS): too advanced. not covered here.

**Example**: replacing a HAL function with a stub.

```c
// Production HAL
uint8_t HAL_ReadRotarySwitch(void) {
    return (uint8_t)(GPIO_ReadPin(ROTARY_PIN));
}

// Stub implementation for SIL
uint8_t HAL_ReadRotarySwitch(void) {
    // Return simulated input value provided by CANoe
    return simulated_switch_position;
}
```

In the stub implementation, the variable `simulated_switch_position` can be updated by the SIL adapter based on CANoe stimuli.

### Build SIL Adapter

The SIL adapter defines what data CANoe can send/receive from the SUT. This is often described in a **vCDL** file (Vector Communication Description Language), which specifies

* provided data: outputs from the SUT (e.g. actuator states)
* consumed data: inputs to the SUT (e.g. sensor readings)

On Windows, SIL adapter is typically built as a dynamically linked library (`.dll` file), which is loaded by CANoe at runtime. It allows CANoe to stimulate inputs and observe outputs without physical hardware. For a simple SUT (e.g., sensor $\to$ compute $\to$ actuator), we do not need a full virtual ECU or AUTOSAR stack. Instead, the **CANoe SIL Adapter Builder** can generate a lightweight adapter automatically.

**Example**: PowerShell command to build the SIL adapter.

```shell
/path/to/sil-adapter-builder.exe /path/to/myProj.vCDL -o . -l cpp
```

Question:

1. example cVDL file
1. next step?
