---
title: "SIL Test"
date: "2025"
author: "Kezhang"
---

# Software in the Loop Test

Consider an embedded software running on a microprocessor(uC) that connects to multiple peripherals. In unit tests, we test the software units in virtual environments such as virtual machine or docker container.

In integration test, we would like to test the overall system or a subsystem. How to test the the system without the physical hardware? We need some virual environment that does no only simulates the target uC, but also the topology between the uC and its peripherals. This is exactly the basic idea of **software-in-the-loop (SIL) testing**.

CANoe is the common tool for SIL testing. We can build the simulation environment inside CANoe that mimics the structure of the real system.

A **SIL adapter** is used to connect the compiled software (typically a `.hex` or `.axf` file) to the simulation environment. (Question: is the compiled software the so called system under test?) It serves as the bridge between the software and CANoe.

Basic workflow of SIL test:

1. Preparation:
    * install CANoe (with SIL adapter support)
    * install CMake or Visual Studio (for building the SIL adapter)
    * build the software binaries (`.hex` or `.axf`)
    * ensure the software is runnable in uC emulator
1. Create SIL adapter
    * define the interfaces in CANoe
    * map the interfaces to software functions
    * bulid the adapter using CMake or Visual Studio
1. Connect the SIL adapter to the software
1. Create CANoe simulation environment
    * create network simulation (CAN, LIN, Ethenet, etc.)
    * add peripherals (sensors, actors)
1. Connect the SIL adapter to the simulation environment
1. Define test cases in CAPL
1. Run and debug the system

Example: embedded software running on an arm processor, which connects to a rotary switch and an LED. As the state of rotary switch is changed, the processor sets a different blick pattern to the LED.