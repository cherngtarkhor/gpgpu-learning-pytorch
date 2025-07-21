#!/bin/bash

# Run the script with chmod +x repeat_test.sh
# $0 is the name of the script itself
# $? is the exit status of the last command that was executed

## Accept user input for numbers to repeat the test
if [ "$#" -ne 4 ]; then
    echo "!!!Error: Different Number of Arguments!!!"
    echo "Example Flags: [--core | --tutorial | --unit | --microbench | --softmax | --gemm | --attention | --venv | --skip-pip-install | --skip-pytorch-install | --reports | --reports-dir DIR | --warning-reports | --ignore-errors | --skip-list SKIPLIST]" 
    exit 1
fi

VENV=$1
TEST_DIR=$2
MAX_NUMBER=$3
FLAG=$4

export MAX_JOBS=16
source $TEST_DIR/$VENV/bin/activate
#source /opt/intel/oneapi/2025.0/oneapi-vars.sh
#cd $TEST_DIR

#Check Log File in TritonXPU repo
if [ -e "./xpu_backend/log_file" ]; then
    echo "Log File Exists..."
else
    echo "Creating Log File Directory..."
    mkdir ./xpu_backend/log_file
fi

#Check Log File in TritonXPU repo
if [ -e "./xpu_backend/reports" ]; then
    echo "Reports Dir Exists..."
else
    echo "Creating Report Directory..."
    mkdir ./xpu_backend/reports
fi

#Check Output File in TritonXPU repo
if [ -e "./xpu_backend/outputs" ]; then 
    echo "Output File Exists..."
else
    echo "Creating output.csv File..."
    mkdir ./xpu_backend/outputs
fi

for ((i=1; i<=MAX_NUMBER; i++)); do
    # Get the current date and time in YYYY-MM-DD_HH-MM-SS format
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    LOG_FILE="./xpu_backend/log_file/test-log_${TIMESTAMP}.txt"
    REPORT="./xpu_backend/reports/report_${TIMESTAMP}"
    CSV_FILE="./xpu_backend/outputs/output_${TIMESTAMP}.csv"
    
    {
        echo "Time: $TIMESTAMP"
        echo "Number of Test Repeat: $MAX_NUMBER"
        echo "Test Path: $TEST_DIR"
        echo "Flag used: $FLAG"
        echo "Environment: $VENV"
        echo "Initiating $i run of test-triton.sh"
        echo " "
    } |& tee -a "$LOG_FILE"

    #Run Test-Triton Script
    $TEST_DIR/scripts/test-triton.sh $FLAG |& tee -a "$LOG_FILE"
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "test-triton.sh run completed successfully with return 0." |& tee -a "$LOG_FILE"

        touch $CSV_FILE
        echo "Test Type,Tests Passed,Tests Failed,Skipped,xfailed,Total Test,Warnings,Total Time (s)" > "$CSV_FILE"
        echo "$TIMESTAMP" >> "$CSV_FILE"
        COMMIT=$(git rev-parse HEAD)
        echo "CommitID:${COMMIT:0:7}" >> "$CSV_FILE"
        echo "Environment:$VENV" >> "$CSV_FILE"
        echo "($FLAG)" >> "$CSV_FILE"
        
        # Extract Triton CXX test results
        if grep -q "Running Triton CXX unittests" "$LOG_FILE"; then                 #100% tests passed, 0 tests failed out of 161
            PASSED=$(grep "tests passed" "$LOG_FILE" | awk '{gsub("%", ""); print $NF}')
            FAILED=$(grep "tests failed" "$LOG_FILE" | awk '{print $4}')
            SKIPPED=0
            XFAILED=0
            TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
            WARNINGS=0
            TIME=$(grep "Total Test time" "$LOG_FILE" | awk '{print $(NF-1)}')
            echo "CXX,$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
        fi

        # Extract Triton LIT test results
        if grep -q "Running Triton LIT tests" "$LOG_FILE"; then
            PASSED=$(grep "Passed:" "$LOG_FILE" | awk '{print $2}')
            FAILED=0
            SKIPPED=0
            XFAILED=0
            TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
            WARNINGS=0
            TIME=$(grep "Testing Time:" "$LOG_FILE" | awk '{gsub("s", ""); print $(NF)}')
            echo "LIT,$PASSED,$(($TOTAL - $PASSED)),$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
        fi

        # Extract Triton Core test results
        if grep -q "Running Triton Core tests" "$LOG_FILE"; then

            # Language
            VAR=$(grep ".* failed, .* passed, .* skipped, .* xfailed, .* warnings" "$LOG_FILE")
            PASSED=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
            FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
            SKIPPED=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
            XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $7}')
            TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
            WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $9}')
            TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF-1)}')
            echo "CORE(Language),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
            
            # Subprocess
            if grep -Eq '\[100%\] PASSED language/test_subprocess' "$LOG_FILE"; then

                if grep -oPq '=\s*\d+ failed, \d+ passed in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ failed, \d+ passed in \d+\.\d+s' "$LOG_FILE")
                    PASSED=$(echo "$VAR" | awk '{gsub("=", "");print $3}')
                    FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                    SKIPPED=0
                    XFAILED=0
                    TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                    WARNINGS=0
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                    echo "CORE(Subprocess),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"

                elif grep -oPq '=\s*\d+ passed in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ passed in \d+\.\d+s' "$LOG_FILE")
                    PASSED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                    FAILED=0
                    SKIPPED=0
                    XFAILED=0
                    TOTAL=$((PASSED+FAILED+SKIPPED+XFAILED))
                    WARNINGS=0
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                    echo "CORE(Subprocess),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                fi
            else
                PASSED=$(echo "Run Fail")
                echo "CORE(Subprocess),$PASSED" >> "$CSV_FILE"
            fi

            #Runtime
            if grep -Eq "runtime/test.*PASSED \[100%\]" "$LOG_FILE"; then

                if grep -oPq '=\s*\d+ failed, \d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ failed, \d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | head -n 1)
                    PASSED=$(echo "$VAR" | awk '{gsub("=", "");print $3}')
                    FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                    SKIPPED=0
                    XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
                    TOTAL=$((PASSED+FAILED+SKIPPED+XFAILED))
                    WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $7}')
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                    echo "CORE(Runtime),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                
                elif grep -oPq '=\s*\d+ passed, \d+ deselected, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ passed, \d+ deselected, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | head -n 1)
                    PASSED=$(echo "$VAR" | awk '{gsub("=", "");print $1}')
                    FAILED=0
                    SKIPPED=0
                    XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
                    TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                    WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $7}')
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                    echo $VAR
                    echo "CORE(Runtime),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                fi
            else
                PASSED=$(echo "Run Fail")
                echo "CORE(Runtime),$PASSED" >> "$CSV_FILE"
            fi

            # Debug
            if grep '\[100%\] PASSED test_debug' "$LOG_FILE"; then

                if grep -oPq '=\s*\d+ passed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ passed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | head -n 1)
                    PASSED=$(echo "$VAR" | awk '{gsub("=", "");print $1}')
                    FAILED=0
                    SKIPPED=0
                    XFAILED=0
                    TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                    WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                    echo "CORE(Debug),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                fi
            else
                PASSED=$(echo "Run Fail")
                echo "CORE(Debug),$PASSED" >> "$CSV_FILE"
            fi
            
            # Line_Info
            if grep -Eq "language/test_line_info.*PASSED\s*\[100%\]" "$LOG_FILE"; then

                if grep -oPq '=\s*\d+ passed, \d+ deselected in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ passed, \d+ deselected in \d+\.\d+s' "$LOG_FILE")
                    PASSED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                    FAILED=0
                    SKIPPED=0
                    XFAILED=0
                    TOTAL=$(($PASSED + $FAILED + $SKIPPED + $XFAILED))
                    WARNINGS=0
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}')
                    echo "CORE(Line_Info),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                fi
            else
                PASSED=$(echo "Run Fail")
                echo "CORE(Line_Info),$PASSED" >> "$CSV_FILE"
            fi

            # Tools
            if grep -Eq "tools/test.*PASSED.*\[100%\]" "$LOG_FILE"; then

                if grep -oPq '=\s*\d+ failed, \d+ passed, \d+ deselected in \d+\.\d+s' "$LOG_FILE"; then
                    VAR=$(grep -oP '=\s*\d+ failed, \d+ passed, \d+ deselected in \d+\.\d+s' "$LOG_FILE")
                    PASSED=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
                    FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                    SKIPPED=0
                    XFAILED=0
                    TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                    WARNINGS=0
                    TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}')
                    echo $VAR
                    echo "CORE(Tools),$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                fi 
            else
                PASSED=$(echo "Run Fail")
                echo "CORE(Tools),$PASSED" >> "$CSV_FILE"
            fi

           # Missing "Third Party"
        fi

        if grep -q "Running Triton Regression tests" "$LOG_FILE"; then

            if grep -oPq '=\s*\d+ failed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                VAR=$(grep -oP '=\s*\d+ failed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | tail -n 1)
                PASSED=0
                FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                SKIPPED=0
                XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
                TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
                TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                echo "Regression,$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"

            elif grep -oPq '=\s*\d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                VAR=$(grep -oP '=\s*\d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | tail -n 1)
                PASSED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                FAILED=0
                SKIPPED=0
                XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
                TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
                TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                echo "Regression,$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
                
            elif grep -oPq '=\s*\d+ failed, \d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE"; then
                VAR=$(grep -oP '=\s*\d+ failed, \d+ passed, \d+ xfailed, \d+ warnings in \d+\.\d+s' "$LOG_FILE" | tail -n 1)
                PASSED=$(echo "$VAR" | awk '{gsub("=", "");print $3}')
                FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                SKIPPED=0
                XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $5}')
                TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                WARNINGS=$(echo "$VAR" | awk '{gsub("=", ""); print $7}')
                TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                echo "Regression,$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"

            elif grep -oPq '=\s*\d+ failed, \d+ xfailed, in \d+\.\d+s' "$LOG_FILE"; then
                VAR=$(grep -oP '=\s*\d+ failed, \d+ xfailed, in \d+\.\d+s' "$LOG_FILE" | tail -n 1)
                PASSED=0
                FAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $1}')
                SKIPPED=0
                XFAILED=$(echo "$VAR" | awk '{gsub("=", ""); print $3}')
                TOTAL=$(($PASSED+$FAILED+$SKIPPED+$XFAILED))
                WARNINGS=0
                TIME=$(echo "$VAR" | awk '{gsub("[=s]", ""); print $(NF)}' | tr -d '()')
                echo "Regression,$PASSED,$FAILED,$SKIPPED,$XFAILED,$TOTAL,$WARNINGS,$TIME" >> "$CSV_FILE"
            fi
        fi

        if grep -q "Running Triton Tutorial tests" "$LOG_FILE"; then
            newest_file=$(ls -t ${HOME}/reports | sort -r | head -n 1)
            cp -r "${HOME}/reports/${newest_file}" "./xpu_backend/reports/report_${TIMESTAMP}"
            tutorials=("01" "02" "03" "04" "05" "06" "07" "08" "10" "10i")

            for j in "${tutorials[@]}"; do
                file="./xpu_backend/reports/report_${TIMESTAMP}/tutorial-${j}-*.txt"
                result=$(cat $file)
                echo $file
                echo $result

                # Determine the result based on file content
                if [[ $result == *"PASS"* ]]; then
                    echo "Tutorial $j,PASS" >> "$CSV_FILE"

                elif [[ $result == *"FAIL"* ]]; then
                    echo "Tutorial $j,FAIL" >> "$CSV_FILE"

                elif [[ $result == *"SKIP"* ]]; then
                    echo "Tutorial $j,SKIP" >> "$CSV_FILE"

                else
                    echo "Tutorial $j,UNKNOWN" >> "$CSV_FILE"
                fi
            done
        fi

        if [ $? -eq 0 ]; then
            echo "Come into Git@@@@@@"

            git fetch origin

            git merge origin/main
            
            # Add the specified files to the staging area
            git add "$CSV_FILE" "$LOG_FILE" "$REPORT" 

            # Commit the changes with a descriptive message
            git commit -m "Uploaded ${LOG_FILE}, ${CSV_FILE}, and ${REPORT}"

            # Push the changes to the remote repository
            git push origin main
        else
            echo "Merge failed. Please resolve any conflicts."
        fi

    else
        echo "test-triton.sh run failed with non-zero code returned." |& tee -a "$LOG_FILE"
    fi
    sleep 5
done

echo ""
echo "Test Finished!"
