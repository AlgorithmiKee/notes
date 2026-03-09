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

* SIL is used for integration and system-level testing, which is a higher level of verification than unit testing. Compared to hardware-in-the-loop (HIL), SIL does not require physical hardware.
* The application software runs natively on the host PC, not on the target machine.
  * Hardware-dependent components are replaced by stubs that preserve their interface but do not replicate the actual hardware behavior.
  * The external environment can be simulated by tools such as CANoe, but other simulation tools may also be used depending on the project requirements.

A **stub** is a host-executable implementation that replaces a hardware-dependent component while preserving its interface.

* Hardware-dependent components are typically HAL functions.
* This allows the application software to execute in a hardware-free environment.
* The stub implementation exposes sensor reading and actuator status as simulation variables (often global), which are updated by the SIL adapter and read by the SUT.

The **system under test (SUT)** is a host-compiled variant of the application software, typically built as a dynamic library (e.g. `.dll` on Windows).

* The SUT is compiled to run natively on the host PC, not cross-compiled for the target machine.
* In SIL, all hardware-dependent components in SUT are replaced with stubs.
* The SUT exposes function interfaces to external parties (e.g. `main.cpp` of test harness). Those interfaces typically include SUT initialization, SUT step, and data exchange functions. The test harness calls those interfaces to execute the SUT.

A **SIL adapter** is a dynamically loaded library (DLL) that connects CANoe with the SUT. It handles data exchange between CANoe and the SUT.

The **test harness** is the final executable that runs on the host PC during test execution. It loads the SUT dll and the SIL adapter, and manages the test execution flow.

> Note: In some contexts, the term 'SUT' may refer to the test harness, which includes both the SUT dll and the SIL adapter. However, in this document, we use 'SUT' to refer specifically to the control logic of the application software, while the 'test harness' refers to the entire executable that includes both the SUT and the SIL adapter.

## Overview of SIL Test Workflow

In the following, we assume Windows as the host OS, and CANoe as the simulation tool. The general workflow is similar for other OS and simulation tools, but the details may differ.

How to build a SIL test?

1. Preparation:
    * install CANoe (with SIL adapter support)
    * install CMake and compiler toolchain (for building the SIL adapter and SUT)
1. Build the SUT for SIL
    * replace the hardware-dependent modules with stubs.
    * build the SUT as windows dll.
1. Create a new CANoe project for SIL testing
1. Generate source files for SIL adapter
    * define the communication interfaces in a `.vCDL` file
    * generate the source files for the SIL adapter using CANoe SIL Adapter Builder
1. Build the test harness
    * create a main.cpp that serves as the entry point of the test harness. It typically includes code for loading the SUT dll, initializing the SIL adapter, bridging the communication between CANoe and SUT, and executing the test cases.
    * link the SUT dll and the SIL adapter dll together to create the test harness executable
1. Optional: create a panel for graphical stimulating/monitoring of the SUT outputs in CANoe

How to run the SIL test manually?

1. Open the SIL test project in CANoe and start simulation.
1. Launch the test harness executable on the host PC.
1. Stimulate the SUT by sending signals from CANoe. This can be done either by using the controls in the CANoe panel, or by modifying the CANoe variables directly.
1. Monitor the SUT outputs.

## Step-by-Step Guide

Example: embedded software running on an arm processor, which connects to a rotary switch and an LED. As the switch position changes, the microcontroller updates the LED blink pattern.

### Build the SUT for SIL Test

Prepare the code for SIL:

* sensor and actuator status are typically modelded as global variables.
* interrupt-driven behavior is typically modeled using scheduled callbacks or polling mechanisms.

**Example**: stub for rotary switch (sensor)

```c
// rotary_sw.h  (common header for APIs)
uint8_t read_rotary_sw(void);

// rotary_sw.c  (production code)
#include "rotary_sw.h"                  // common header for APIs
#include "stm32xx.h"                    // hardware specific header
uint8_t read_rotary_sw(void) {
    return (uint8_t)(HAL_ReadPin(ROTARY_PIN));
}

// rotary_sw_stub.c  (SIL test code)
#include "rotary_sw.h"                  // common header for APIs
#include "sil_sim_data.h"               // contains virtual hardware states

uint8_t read_rotary_sw(void) {
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
