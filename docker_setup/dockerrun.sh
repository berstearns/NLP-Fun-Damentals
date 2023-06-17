#!/bin/bash
docker run -it\
	   -v $(PWD)/general_experiments:/app/general_experiments/\
	   -v $(PWD)/submissions:/app/submissions/\
	   phd-experiments bash
