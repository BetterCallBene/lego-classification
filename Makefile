.PHONY: build activate clean package_testing package_stable help

AR_DEV_ARPM_BASE := $(shell PYTHONPATH=""; eval `arpm_env --no-update "ar-dev/[^3.0]@ar/stable"`; echo $$AR_DEV_ARPM_BASE)
include $(AR_DEV_ARPM_BASE)/make/ar-dev.mk

build:
	mkdir -p build
	conan install -g virtualenv -g cmake -if build/ .

activate:
	. build/activate.sh 

clean:
	rm -rf build

package_testing:
	conan create . ${USER}/testing

package_stable:
	conan create . ar/stable

test:
	pytest

help:
	@echo "Available Targets:"
	@echo "... build"
	@echo "... activate"
	@echo "... clean"
	@echo "... package_testing"
	@echo "... package_stable"
	@echo "... help"
