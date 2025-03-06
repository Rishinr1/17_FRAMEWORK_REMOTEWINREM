# Define parameters
$machines = Get-Content "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\machine_list.txt"  # List of target machines
$pythonInstaller = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
$tempDir = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp"
$scriptPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\sample.py"
$configPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\config.ini"
$csvPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\NZEQ_modifiers_2_combinations.csv"
$requirementsPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\requirements.txt"
$sharedDir = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK"
$logPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\setup_log.txt"

$ErrorActionPreference = 'Stop'




# Logging function
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] $Message"
    Write-Host $logEntry
    Add-Content -Path $logPath -Value $logEntry
}


# Test remote machine access
function Test-RemoteMachineAccess {
    param([string]$Machine)
    try {
        $result = Invoke-Command -ComputerName $Machine -ScriptBlock { 
            return $env:COMPUTERNAME 
        } -ErrorAction Stop
        return $true
    }
    catch {
        Write-Log "Cannot access machine $Machine. Error: $($_.Exception.Message)"
        return $false
    }
}

# Ensure directories exist
function Ensure-Directory {
    param([string]$Path)
    if (!(Test-Path -Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Log "Created directory: $Path"
    }
}

# Comprehensive Python Installation Function
function Install-PythonAndSetup {
    param(
        [string]$Machine, 
        [string]$PythonInstaller, 
        [string]$TempDir, 
        [string]$RequirementsPath
    )

    try {
        # Ensure temp directory exists on remote machine
        $remoteTemp = "\\$Machine\$($TempDir.Replace(':', '$'))"
        Ensure-Directory -Path $remoteTemp

        # Create local Python folders
        $localVenvPath = "C:\Python_Venv_$Machine"
        
        # Comprehensive Python and Virtual Environment Setup
        $venvSetup = @"
`$ErrorActionPreference = 'Stop'

# Detect Python installation
`$pythonExe = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source

# If no Python found, use potential default paths
if (`$null -eq `$pythonExe) {
    `$potentialPaths = @(
        'C:\Python311\python.exe', 
        'C:\Python\python.exe', 
        'C:\Program Files\Python311\python.exe'
    )
    
    foreach (`$path in `$potentialPaths) {
        if (Test-Path `$path) {
            `$pythonExe = `$path
            break
        }
    }
}

# Verify Python executable
if (`$null -eq `$pythonExe -or !(Test-Path `$pythonExe)) {
    throw "No Python executable found. Please install Python manually."
}

# Create virtual environment
& "`$pythonExe" -m venv "$localVenvPath"

# Activate virtual environment
`$venvActivate = Join-Path "$localVenvPath" "Scripts\Activate.ps1"
& `$venvActivate

# Use virtual environment Python
`$venvPython = Join-Path "$localVenvPath" "Scripts\python.exe"

# Upgrade pip and install requirements
& "`$venvPython" -m pip install --upgrade pip
& "`$venvPython" -m pip install -r "$RequirementsPath"

# Optional: Test import of libraries
& "`$venvPython" -c "import pandas; print('Pandas version:', pandas.__version__)"
"@
        
        # Execute setup on remote machine
        $venvResult = Invoke-Command -ComputerName $Machine -ScriptBlock { 
            param($cmd) 
            try {
                $output = Invoke-Expression $cmd 2>&1
                return @{Success = $true; Output = $output}
            } catch {
                return @{Success = $false; Error = $_.Exception.Message}
            }
        } -ArgumentList $venvSetup

        if ($venvResult.Success) {
            Write-Log "Python virtual environment setup completed on $Machine"
            Write-Log "Output: $($venvResult.Output)"
            return $true
        } else {
            Write-Log "Error setting up Python virtual environment on $Machine`: $($venvResult.Error)"
            return $false
        }
    }
    catch {
        $errorMessage = $_.Exception.Message
        $errorTrace = $_.ScriptStackTrace
        Write-Log "Error setting up Python on $Machine`: $errorMessage"
        Write-Log "Error Trace: $errorTrace"
        return $false
    }
}


# Distribute CSV rows
function Distribute-CSVRows {
    param(
        [string[]]$Machines,
        [string]$CsvPath
    )

    try {
        $csvData = Import-Csv $CsvPath
        $totalRows = $csvData.Count
        $rowsPerMachine = [math]::Ceiling($totalRows / $Machines.Count)
        $index = 0

        Write-Log "Distributing $totalRows rows across $($Machines.Count) machines"

        foreach ($machine in $Machines) {
            $machineDir = "\\$machine\D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp\"
            $csvSplitPath = "$machineDir\split_data_$machine.csv"
            
            # Ensure directory exists
            Ensure-Directory -Path $machineDir
            
            # Select rows for this machine
            $machineRows = $csvData | Select-Object -Skip $index -First $rowsPerMachine
            $index += $rowsPerMachine
            
            # Export machine-specific CSV
            $machineRows | Export-Csv -Path $csvSplitPath -NoTypeInformation
            Write-Log "Distributed $($machineRows.Count) CSV rows for $machine to $csvSplitPath"
        }
        return $true
    }
    catch {
        Write-Log "Error distributing CSV rows: $($_.ToString())"
        return $false
    }
}

# Execute script on machines
function Execute-ScriptOnMachines {
    param(
        [string[]]$Machines,
        [string]$ScriptPath,
        [string]$ConfigPath
    )

    $jobs = @()
    foreach ($machine in $Machines) {
        $localVenvPath = "\\$machine\D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp\\Python_Venv_$machine"
        $csvSplitPath = "\\$machine\D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp\split_data_$machine.csv"
        $localScriptPath = "$scriptPath"
        $localConfigPath = "$configPath"
        
        $executeCmd = @"
`$ErrorActionPreference = 'Stop'

# Detect Python in virtual environment
`$pythonExe = Join-Path "$localVenvPath" "Scripts\python.exe"

# If virtual env Python not found, fall back to system Python
if (!(Test-Path "`$pythonExe")) {
    `$pythonExe = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
}

# Verify Python executable exists
if (`$null -eq `$pythonExe -or !(Test-Path `$pythonExe)) {
    throw "No Python executable found for script execution"
}

# Execute script
& "`$pythonExe" "$localScriptPath" --input "$csvSplitPath" --config "$localConfigPath"
if (`$LASTEXITCODE -ne 0) {
    throw "Script execution failed with exit code `$LASTEXITCODE"
}
"@
        
        $jobs += Start-Job -ScriptBlock {
            param($machine, $executeCmd)
            try {
                $result = Invoke-Command -ComputerName $machine -ScriptBlock { 
                    param($cmd)
                    try {
                        $output = Invoke-Expression $cmd 2>&1
                        return @{Success = $true; Output = $output}
                    } catch {
                        return @{Success = $false; Error = $_.Exception.Message}
                    }
                } -ArgumentList $executeCmd
                
                if ($result.Success) {
                    return "Script executed successfully on $machine. Output: $($result.Output)"
                } else {
                    return "Error executing script on $machine`: $($result.Error)"
                }
            }
            catch {
                return "Error executing script on $machine`: $($_.ToString())"
            }
        } -ArgumentList $machine, $executeCmd
    }

    # Wait and collect results
    $results = $jobs | Wait-Job | Receive-Job
    $results | ForEach-Object { Write-Log $_ }
}

# Main Execution Flow
try {
    # Create log file
    if (!(Test-Path -Path $logPath)) {
        New-Item -ItemType File -Path $logPath -Force | Out-Null
    }
    
    Write-Log "=== Starting distributed processing setup ==="
    Write-Log "Target machines: $($machines -join ', ')"
    
    # Check machine accessibility
    $accessibleMachines = $machines | Where-Object { Test-RemoteMachineAccess -Machine $_ }
    if ($accessibleMachines.Count -eq 0) {
        throw "No machines are accessible for distributed processing"
    }
    Write-Log "Accessible machines: $($accessibleMachines -join ', ')"
    
    # Ensure main directories
    Ensure-Directory -Path $tempDir
    Ensure-Directory -Path $sharedDir

    # Install Python and setup environments
    Write-Log "Setting up Python environments on target machines..."
    $pythonSetupResults = @()
    foreach ($machine in $accessibleMachines) {
        Write-Log "Starting Python setup on $machine"
        $result = Install-PythonAndSetup -Machine $machine -PythonInstaller $pythonInstaller -TempDir $tempDir -RequirementsPath $requirementsPath
        $pythonSetupResults += $result
        Write-Log "Python setup on $machine completed with result: $result"
    }

    if ($pythonSetupResults -contains $false) {
        Write-Log "WARNING: Python setup failed on one or more machines, proceeding anyway"
    }

    # Distribute CSV rows
    Write-Log "Distributing CSV data to machines..."
    $csvDistributionResult = Distribute-CSVRows -Machines $accessibleMachines -CsvPath $csvPath
    
    if (-not $csvDistributionResult) {
        throw "Failed to distribute CSV data"
    }

    # Execute script on machines
    Write-Log "Executing script on machines..."
    Execute-ScriptOnMachines -Machines $accessibleMachines -ScriptPath $scriptPath -ConfigPath $configPath

    Write-Log "Distributed processing completed"
}
catch {
    $errorMessage = $_.Exception.Message
    $errorTrace = $_.ScriptStackTrace
    Write-Log "Critical error in distributed processing: $errorMessage"
    Write-Log "Error Trace: $errorTrace"
}
finally {
    # Cleanup jobs
    Write-Log "Cleaning up background jobs"
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
}