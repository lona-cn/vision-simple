option("with_dml")
set_showmenu(true)
--set_default(true)
if is_os("windows") then
set_default(true)
else
set_default(false)
end
add_defines("VISION_SIMPLE_WITH_DML")
set_description("ONNXRuntime with DirectML Execution Provider")
option_end()