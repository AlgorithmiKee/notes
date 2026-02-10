---
title: "SIL Test"
date: "2025"
author: "Ke Zhang"
---

Proof-read my notes about SIL test, which is for me a quite new topic. There might be misunderstanding of the subject. Read it carefully and make sure everthing is written precisely and rigorously. Address the inline questions and TODOs.

# Software in the Loop Test

## Basic Terms

The **host** machine (or host PC) refers to the computer where the development work is done. It is typically a Windows or Linux laptop.

The **target machine** refers to the chip where the application software is run. It is typically an ARM microcontroller.

**Software-in-the-Loop (SIL)** testing is a verification method where the application software code is tested within a simulated environment that mimics the hardware and external systems.

* SIL is typically used for integration testing and early system-level testing. Compared to unit test, SIL is a higher level of verification. Compared to hardware-in-the-loop (HIL), SIL does not require physical hardware.
* The application software runs natively on the host PC, not on the target machine.
  * Hardware-dependent components are replace by stubs.
  * External environment is simulated by CANoe.

A **stub** is a host-executable implementation that replaces a hardware-dependent component while preserving its interface.

* Hardware-dependent components are typically HAL functions.
* This allows the application software to execute in a hardware-free environment.

The **system under test (SUT)** is a host-compiled variant of the application software, built for execution on the host PC instead of the target machine.

* The SUT is not cross-compiled for target machine on the host PC. It is native to the host.
* In SIL, all hardware-dependent components in SUT are replaced with stubs.

A **SIL adapter** is a dynamically loaded library (DLL) that connects CANoe with the SUT. It handles data exchange between CANoe and the SUT.

## Execution Flow

During test execution:

1. CANoe loads the SIL adapter.
1. SIL adapter loads and runs the SUT.
1. CANoe sends test stimuli into SIL adapter.
1. SIL adapter translates the test stimuli into inputs understood by the SUT (e.g. variable udpates or function calls).
1. The SUT receives the input from the SIL adapter, processes it, and reacts to it by sending outputs.
1. The SIL adapter converts the SUT outputs back to CANoe for monitoring and evaluation.

## Step-by-Step Guide

Example: embedded software running on an arm processor, which connects to a rotary switch and an LED. As the switch position changes, the microcontroller updates the LED blink pattern.

Basic workflow of SIL test:

1. Preparation:
    * install CANoe (with SIL adapter support)
    * install CMake or Visual Studio (for building the SIL adapter)
1. Build the SUT for SIL
    * replace the hardware-dependent modules with stubs.
    * build the software binaries (aka SUT) for SIL test
    * ensure the software is runnable on the host machine
1. Build SIL adapter
    * define the communication interfaces in CANoe (in `.vCDL` file)
    * map the test stimuli to variables or function calls in SUT
    * build the adapter using CMake or Visual Studio
1. Connect the SIL adapter to the software
1. Create CANoe simulation environment
    * create network simulation (CAN, LIN, Ethernet, etc.)
    * add simulated peripherals (sensors, actuators)
1. Connect the SIL adapter to the simulation environment
1. Define test cases in CAPL
    * write stimuli, checks, and verdicts (PASS/FAIL)
1. Run and debug the system
    * execute measurements, inspect traces, log failures, and iterate
    * optional: automate testing

### Prepare the Source Code for SIL

The production code typically calls **hardware abstraction layer (HAL)** functions for accessing the uC registers and peripherals. Before SIL test, we must replace those HAL functions with **stub functions** (or simply **stubs**) which simulate the hardware behavior. The SUT is runnable on the host PC without physical hardware. Hence, it must be built by linking against stubs.

Remarks:

* The stubs are also known as **adapter functions** or **mock HALs**. They are simulation-based implementations of the HAL APIs.
* 💣 Not to be confused by the wording *simulation*:
  * Stubs simulate the internal hardware-dependent behaviour of SUT
  * CANoe simulates the external environment which the SUT interacts with
* 💣 Stubs are not tied to CANoe. They are independently runnable on the host PC.

The best practice for writing stubs:

1. keep the HAL API intact for portability
1. provide a simulated implementation of the HAL functions. This can be done either via conditional compilation (e.g. `#ifdef SIL_BUILD`) or separate source files.
1. simulate I/O by reading/writing shared global variables that represent sensor/actuator states.

**Example**: stub for rotary switch (sensor)

```c
// hal.c  (production HAL)
uint8_t HAL_ReadRotarySwitch(void) {
    return (uint8_t)(GPIO_ReadPin(ROTARY_PIN));
}

// sil_stub.c  (stubbed HAL for SIL)
#include "sil_sim_data.h"               // contains virtual hardware states

uint8_t HAL_ReadRotarySwitch(void) {
    return simulated_switch_position;   // declared in sil_sim_data.h
}
```

Remarks:

* `simulated_switch_position` is a global variable representing the simulated sensor state. It is defined in `sil_sim_data.h` and updated by the SIL adapter at runtime.
* The SIL adapter receives stimuli from CANoe (e.g. set the rotary switch to position 2) and maps that stimuli to `simulated_switch_position`.
* The SUT reads `simulated_switch_position` and process it further (e.g. update the blink pattern accordingly).

With all HAL functions being replaced, we can build the binary aka the SUT. The format of SUT depends on the execution strategy:

* host-native build: build the SUT as a Windows/Linux executable or library. On windows, SUT is typically built as `.exe` or `.dll`.
* instruction-set simulation (ISS): too advanced. not covered here.

### Build SIL Adapter

The **SIL adapter** is a compiled shared library (e.g. `.dll` on Windows) that serves as the bridge between CANoe and SUT. Its main responsibilities are to:

* load and execute the SUT binary
* translate test stimuli from CANoe into stub function calls or simulated variable updates inside the SUT
* capture SUT outputs and forward them back to CANoe for monitoring and evaluation.

#### Define Interfaces in vCDL

The data exchange between CANoe and the SUT is defined in a **vCDL** file. This file declares what data items CANoe can provide to or receive from the SUT:

* **provided data**: data provided by CANoe, i.e. inputs to the SUT (e.g. sensor readings)
* **consumed data**: data received by CANoe, i.e. outputs from the SUT (e.g. actuator states)

Modern vCDL (v2.0+) uses **IDL (Interface Definition Language)** to describe these interfaces in a structured way. This file acts as the blueprint for generating the SIL adapter.

**Example**: vCDL for blinking LED with rotary switch

```c++
version 2.0

import module "SilKit"
namespace Blinky
{
    [Binding="SilKit"]
    interface IRotary
    {
        provided data int switch_position;
    }

    [Binding="SilKit"]
    interface ILed
    {
        consumed data bool led_state;
        consumed data double blink_frequency;
    }

    IRotary rotary;
    ILed led;
}
```

Remarks:

* `interface` is conceptually similar to class in C++. The prefix `I` in `IRotary` and `ILed` is a common naming convention, not mandatory in vCDL.
* `IRotary rotary;` creates an instance of the interface, just like creating an object from a class.
* `[Binding="SilKit"]` pecifies that SilKit is the communication middleware between CANoe and the SUT.
* Each `provided` or `consumed` data item is like a one-way transmission portal from CANoe to SUT or vice versa.
* For our toy example, a single-interface vCDL also works. $\to$ See [appendix](#single-interface-vcdl).

#### Generate Source Files for SIL Adapter

The **CANoe SIL Adapter Builder** is a Vector tool that reads a `.vCDL` file and generates source code for a SIL adapter in a chosen programming language (typically C++). It automates the creation of code template for

* connecting CANoe to the SUT
* mapping CANoe stimuli to stubs
* handling data exchange during test execution

⚠️ Important: The Adapter Builder does not compile the SIL adpater itself. It only generates the source files for the SIL adapter, which must then be compiled manually into `.dll`.

💡 Good to know: For a simple SUT (e.g., sensor $\to$ compute $\to$ actuator), the code template generated by CANoe SIL Adapter Builder is usually sufficient for building the SIL adapter. A full virtual ECU or AUTOSAR stack is not required.

The CANoe SIL Adapter Builder can be invoked either in the GUI of CANoe, or via the following command line in power shell:

```shell
/path/to/sil-adapter-builder.exe /path/to/myProj.vCDL -o . -l cpp
```

This command typically creates a SilAdapter folder containing

```txt
/SilAdapter/
│
├── RotaryInterface.h
├── RotaryInterface.cpp
├── LedInterface.h
├── LedInterface.cpp
├── main.cpp        ← Adapter entry point
└── CMakeLists.txt  ← For building the adapter
```

Remarks:

* Each interface defined in `.vCDL` file yields a corresponding header-source pair. e.g. The `IRotary` interface yields `RotaryInterface.h` and `RotaryInterface.cpp`.
* We need to implement the bridge logic, not in `*Interface.cpp`, but in `*Interface_User.cpp`. Otherwise, every `.vCDL` file update re-generates and overwrites our implementation in `*Interface.cpp`.

`RotaryInterface.h` would look like

```c++
//------------------------------------------------------------------------------
//  RotaryInterface.h
//  Auto-generated by CANoe SIL Adapter Builder
//------------------------------------------------------------------------------

// Called when CANoe provides a new rotary switch position value.
void RotaryInterface_OnSwitchPositionProvided(int switch_position);

// Called during adapter initialization (optional, empty by default)
void RotaryInterface_Initialize(void);

// Called during adapter shutdown (optional, empty by default)
void RotaryInterface_Terminate(void);
```

`RotaryInterface_OnSwitchPositionProvided` is an auto-generated callback that is triggered whenever CANoe sends a new value for the data item `switch_position`.
Our implementation in the `_User.cpp` file defines what happens when this occurs—typically updating a simulated hardware variable shared with the SUT.

`RotaryInterface.cpp` would look like

```c++
//------------------------------------------------------------------------------
//  RotaryInterface.cpp
//  Auto-generated by CANoe SIL Adapter Builder
//  DO NOT EDIT THIS FILE DIRECTLY
//  Implement your custom logic in RotaryInterface_User.cpp
//------------------------------------------------------------------------------

#include "RotaryInterface.h"

// These default implementations are placeholders.
// Override them in a corresponding "_User.cpp" file.

void RotaryInterface_OnSwitchPositionProvided(int switch_position) {
    // Implement the bridge logic in RotaryInterface_User.cpp
}

void RotaryInterface_Initialize(void) {}
void RotaryInterface_Terminate(void) {}
```

Likewise, for interface `ILed`, the auto-generated files: TODO.

#### Mapping the CANoe Signals to Simulated Values

To implement the bridge logic, create a customized `RotaryInterface_User.cpp` file:

```c++
// RotaryInterface_User.cpp
#include "RotaryInterface.h"
#include "sil_sim_data.h"  // contains simulation data (also used by SUT)

void RotaryInterface_OnSwitchPositionProvided(int switch_position)
{
    simulated_switch_position = switch_position;
}
```

## Appendix

### Single-Interface vCDL

```c++
version 2.0

import module "SilKit"
namespace Blinky
{
    [Binding="SilKit"]
    interface IBlinky_IO
    {   
        // sensors
        provided data int switch_position;

        // actuators
        consumed data bool led_state;
        consumed data double blink_frequency;
    }

    IBlinky_IO blinky_io;
}
```
