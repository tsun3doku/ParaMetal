# ParaMetal

![Heat Transfer](ParaMetal/docs/heat.gif)

This project allows the visualization of heat transfer between two or more 3D closed surface geometry. The simulation is deterministic, transient and operates in realtime using stable pre-processing methods. 

![Voronoi](ParaMetal/docs/voronoi.png)

The major pre-processing methods include an intrinsic remeshing operation that preserves the shape of the geometry and a meshless restricted voronoi diagram step that discretizes the volume of the surface boundary.

![Contact](ParaMetal/docs/contact.png)

This project is a work in progress. Functionality, performance and physical accuracy will be continuously updated.

## Quick Start
1. Download the latest demo release from the [Releases](https://github.com/tsun3doku/ParaMetal/releases) page
2. Extract the zip file
3. Run ParaMetal.exe

### Hardware Requirements
- GPU with Vulkan 1.3 or higher support ([Check GPU compatibility](https://vulkan.gpuinfo.org/))

### Prerequisites
- [CMake](https://cmake.org/download/) 
- [Qt 6](https://www.qt.io/download) (Core, Gui, Widgets components)
- [Vulkan SDK](https://vulkan.lunarg.com/) 1.3 or higher
- [Python 3](https://www.python.org/downloads/) (Development headers required)
- [Pybind11](https://github.com/pybind/pybind11)

### Build Steps For Windows

1. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/tsun3doku/ParaMetal.git
   cd ParaMetal
   ```

2. Configure and build:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH="/YOURPATH/TO/Qt/6.x.x/msvc2022_64"
   cmake --build . --config Release
   ```
   
3. Run the program within build/release 

### Build Steps for Linux (RHEL/CentOS/Fedora)

1. Required packages:
   ```bash
   sudo dnf install cmake gcc-c++ qt6-qtbase-devel vulkan-loader-devel vulkan-headers
   python3-devel pybind11-devel
   ```
2. Clone the repository with submodules:
   ```bash
   git clone --recursive https://github.com/tsun3doku/ParaMetal.git
   cd ParaMetal
   ```

3. Build:
   ```bash
   mkdir build && cd build
   cmake ..
   cmake --build . -j$(nproc)
   ```
   Run:
   ```bash
   ./ParaMetal
   ```
