# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build

# Include any dependencies generated for this target.
include src/caffe/CMakeFiles/caffeproto.dir/depend.make

# Include the progress variables for this target.
include src/caffe/CMakeFiles/caffeproto.dir/progress.make

# Include the compile flags for this target's objects.
include src/caffe/CMakeFiles/caffeproto.dir/flags.make

include/caffe/proto/caffe.pb.cc: ../src/caffe/proto/caffe.proto
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Running C++/Python protocol buffer compiler on /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe/proto/caffe.proto"
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/cmake -E make_directory /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/protoc --cpp_out /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto -I /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe/proto /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe/proto/caffe.proto
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/protoc --python_out /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto -I /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe/proto /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe/proto/caffe.proto

include/caffe/proto/caffe.pb.h: include/caffe/proto/caffe.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate include/caffe/proto/caffe.pb.h

include/caffe/proto/caffe_pb2.py: include/caffe/proto/caffe.pb.cc
	@$(CMAKE_COMMAND) -E touch_nocreate include/caffe/proto/caffe_pb2.py

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o: src/caffe/CMakeFiles/caffeproto.dir/flags.make
src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o: include/caffe/proto/caffe.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o"
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o -c /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto/caffe.pb.cc

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.i"
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto/caffe.pb.cc > CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.i

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.s"
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/include/caffe/proto/caffe.pb.cc -o CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.s

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires:

.PHONY : src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides: src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires
	$(MAKE) -f src/caffe/CMakeFiles/caffeproto.dir/build.make src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides.build
.PHONY : src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides

src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.provides.build: src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o


# Object files for target caffeproto
caffeproto_OBJECTS = \
"CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o"

# External object files for target caffeproto
caffeproto_EXTERNAL_OBJECTS =

lib/libcaffeproto.a: src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o
lib/libcaffeproto.a: src/caffe/CMakeFiles/caffeproto.dir/build.make
lib/libcaffeproto.a: src/caffe/CMakeFiles/caffeproto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library ../../lib/libcaffeproto.a"
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && $(CMAKE_COMMAND) -P CMakeFiles/caffeproto.dir/cmake_clean_target.cmake
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/caffeproto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/caffe/CMakeFiles/caffeproto.dir/build: lib/libcaffeproto.a

.PHONY : src/caffe/CMakeFiles/caffeproto.dir/build

src/caffe/CMakeFiles/caffeproto.dir/requires: src/caffe/CMakeFiles/caffeproto.dir/__/__/include/caffe/proto/caffe.pb.cc.o.requires

.PHONY : src/caffe/CMakeFiles/caffeproto.dir/requires

src/caffe/CMakeFiles/caffeproto.dir/clean:
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe && $(CMAKE_COMMAND) -P CMakeFiles/caffeproto.dir/cmake_clean.cmake
.PHONY : src/caffe/CMakeFiles/caffeproto.dir/clean

src/caffe/CMakeFiles/caffeproto.dir/depend: include/caffe/proto/caffe.pb.cc
src/caffe/CMakeFiles/caffeproto.dir/depend: include/caffe/proto/caffe.pb.h
src/caffe/CMakeFiles/caffeproto.dir/depend: include/caffe/proto/caffe_pb2.py
	cd /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/src/caffe /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe /media/shenruixue/4751501936ECB676/backup/shenruixue/code/face_multiloss/build/src/caffe/CMakeFiles/caffeproto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/caffe/CMakeFiles/caffeproto.dir/depend

