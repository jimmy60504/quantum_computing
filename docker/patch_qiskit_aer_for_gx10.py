from pathlib import Path
import sys


root = Path(sys.argv[1])

pybind = root / "cmake" / "FindPybind11.cmake"
text = pybind.read_text()
text = text.replace(
    """    set_target_properties(${target_name} PROPERTIES\n        CXX_STANDARD 14\n        LINKER_LANGUAGE CXX)\n""",
    """    set_target_properties(${target_name} PROPERTIES\n        CXX_STANDARD 17\n        CUDA_STANDARD 17\n        LINKER_LANGUAGE CXX)\n""",
    1,
)
pybind.write_text(text)

thrust = root / "src" / "simulators" / "statevector" / "chunk" / "thrust_kernels.hpp"
text = thrust.read_text()
text = text.replace(": public thrust::unary_function<difference_type, difference_type>", "", 1)
text = text.replace(
    ": public thrust::unary_function<thrust::complex<data_t>,\n                                    thrust::complex<data_t>>",
    "",
    1,
)
text = text.replace(
    ": public thrust::unary_function<thrust::complex<data_t>,\n                                                    thrust::complex<data_t>>",
    "",
    1,
)
thrust.write_text(text)
