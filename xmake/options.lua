-- dml
if is_os("windows") then
    option("with_dml")
        set_showmenu(true)
        set_default(true)
        add_defines("VISION_SIMPLE_WITH_DML")
        set_description("ONNXRuntime with DirectML Execution Provider")
    option_end()
end

-- cuda & tensorrt
if is_os("windows") or is_os("linux") then
    option("with_cuda")
        set_showmenu(true)
        set_default(false)
        add_defines("VISION_SIMPLE_WITH_CUDA")
        set_description("ONNXRuntime with CUDA Execution Provider")
    option_end()

    option("with_tensorrt")
        set_showmenu(true)
        set_default(false)
        add_defines("VISION_SIMPLE_WITH_TENSORRT")
        set_description("ONNXRuntime with TensorRT Execution Provider")
    option_end()
end