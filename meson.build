project(
  'vulkan_tutorial',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++23'],
)

dependencies = [
  dependency('vulkan', required : true),
  dependency('glfw3', modules: ['glfw']),
]

sources = [
  'vulkan_tutorial.cpp',
]

exe = executable(
  'vulkan_tutorial',
  sources,
  install : true,
  dependencies : dependencies,
)

test('basic', exe)
