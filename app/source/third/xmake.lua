IncludeSubDirs(os.scriptdir())
-- pkgs
add_requires("magic_enum 0.9.6","libhv 1.3.3","turbobase64","yalantinglibs","log4cplus")
-- onnxruntime
if is_arch("x86_64") or is_arch("x64") then
	if has_config("with_dml") then
		add_requires("directml", { system = false })
		add_requires("onnxruntime-dml", { system = false ,alias = "onnxruntime" })
	elseif has_config("with_cuda") or has_config("with_tensorrt") then
		add_requires("onnxruntime", { system = false, configs = { gpu = true } })
	else
		add_requires("onnxruntime", { system = false } )
	end
elseif is_arch("arm64") then	
	if has_config("with_rknpu") then
		add_requires("onnxruntime-git", { system = false, alias = "onnxruntime", configs = { rknpu = true} } )
	else
		add_requires("onnxruntime-git", { system = false, alias = "onnxruntime", configs = { rknpu = false} } )
	end
elseif is_arch("riscv64") then
		add_requires("onnxruntime-git", { system = false, alias = "onnxruntime"} )
end
-- opencv
add_requires("opencv 4.10.0")
--local opencv_enable_cuda = has_config("with_cuda") and true or false
add_requireconfs("opencv", { system = false, configs = { ffmpeg = false, jpeg = true,webp = false,opengl = false,tiff = false,protobuf = false } })
if is_os("linux") then
	add_requireconfs("opencv.python",{ system = true })
	add_requireconfs("yalantinglibs.cinatra.asio",{ system = false})
	if is_arch("arm64") or is_arch("riscv64") then
		add_requireconfs("libhv",{ configs = { defines = "PTHREAD_MUTEX_RECURSIVE", cxflags= "-Wno-implicit-function-declaration" } })
		add_requireconfs("yalantinglibs.cinatra",{configs = {aarch64 = false, ldflags="-lpthread"} })
	end
end
