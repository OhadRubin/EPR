import sys
import shutil

# past command as script parameter
from allennlp.commands import main

sys.argv=sys.argv[1:] # remove script name

serialization_dir = "tmp/debugger_train"

if "train" in sys.argv:
    sys.argv.extend(["-s", serialization_dir])

    # Training will fail if the serialization directory already
    # has stuff in it. If you are running the same training loop
    # over and over again for debugging purposes, it will.
    # Hence we wipe it out in advance.
    # BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
    shutil.rmtree(serialization_dir, ignore_errors=True)

main()