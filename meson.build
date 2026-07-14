project(
  'vulkan_tutorial',
  'cpp',
  version : '0.1',
  meson_version : '>= 1.3.0',
  default_options : ['warning_level=3', 'cpp_std=c++23'],
)

# External Libraries
dependencies = [
  dependency('vulkan', required : true),
  dependency('glfw3', modules: ['glfw']),
  dependency('glm', required : true),
]

# Shader Compilation
slangc = find_program('slangc', required : true)
slang_args = [
 '-target', 'spirv',
 '-profile', 'spirv_1_4',
 '-emit-spirv-directly',
 '-fvk-use-entrypoint-name',
 '-entry', 'vertMain',
 '-entry', 'fragMain',
]
shader_sources = files(
  'shaders/simple.slang',
)
foreach shader : shader_sources
  comp_shader = '@BASENAME@.spv'
  custom_target(
    input : shader,
    output : comp_shader,
    command : [slangc, '@INPUT@', slang_args, '-o', '@OUTPUT@'],
    build_by_default : true,
  )
endforeach

# Source files
sources = [
  'vulkan_tutorial.cpp',
]

executable(
  'vulkan_tutorial',
  sources,
  dependencies : dependencies,
)