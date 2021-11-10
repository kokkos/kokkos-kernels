pipeline {
    agent none

    stages {
        stage('Build & Run') {
	    parallel {
		stage('SYCL-OneAPI') {
		    agent {
			dockerfile {
			    filename 'Dockerfile.sycl'
			    dir 'scripts/docker'
			    label 'nvidia-docker && volta'
			    args '-v /tmp/ccache.kokkos:/tmp/ccache'
			}
		    }
		    steps {
			sh '''rm -rf kokkos &&
			      git clone -b develop https://github.com/kokkos/kokkos.git && cd kokkos && \
			      mkdir build && cd build && \
			      cmake \
				-DCMAKE_BUILD_TYPE=Release \
				-DCMAKE_CXX_COMPILER=clang++ \
				-DKokkos_ARCH_VOLTA70=ON \
				-DKokkos_ENABLE_DEPRECATED_CODE_3=OFF \
				-DKokkos_ENABLE_SYCL=ON \
				-DKokkos_ENABLE_UNSUPPORTED_ARCHS=ON \
				-DCMAKE_CXX_STANDARD=17 \
				.. && \
			      make -j8 && make install && \
			      cd ../.. && rm -rf kokkos'''
			sh '''rm -rf build && mkdir -p build && cd build && \
			      cmake \
				-DCMAKE_BUILD_TYPE=Release \
				-DCMAKE_CXX_COMPILER=clang++ \
				-DKokkosKernels_ENABLE_TESTS=ON \
				-DKokkosKernels_ENABLE_EXAMPLES=ON \
				-DKokkosKernels_INST_DOUBLE=ON \
				-DKokkosKernels_INST_ORDINAL_INT=ON \
				-DKokkosKernels_INST_OFFSET_INT=ON \
			      .. && \
			      make -j8 && ctest --verbose'''
		    }
		}
	    }
        }
    }
}
