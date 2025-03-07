# Define parameters
$machines = Get-Content "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\machine_list.txt"  # List of target machines
$pythonInstaller = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
$baseDir = "D$\RISHIN\framework_for_uni_py\SETUP_WORK"  # Base directory on each machine
$tempDirName = "temp"
$scriptPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\sample.py"
$configPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\config.ini"
$csvPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\NZEQ_modifiers_2_combinations.csv"
$requirementsPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\requirements.txt"
$sharedDir = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK"
$logPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\setup_log.txt"
$venvDirName="venv"

# Logging function
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] $Message"
    Write-Host $logEntry
    Add-Content -Path $logPath -Value $logEntry
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
        [string]$Machine
    )

    try {
        # Create proper machine-specific paths
        $remoteTempDir = "\\$Machine\$baseDir\$tempDirName"
        $remoteRequirementsPath = "\\$Machine\$baseDir\requirements.txt"
        $localTempDir = "D:\RISHIN\framework_for_uni_py\SETUP_WORK\$tempDirName"  # Local path on the remote machine
        $venvPath = "$localTempDir\$venvDirName"  # Virtual environment path (local to the machine)
        
        # Ensure temp directory exists on remote machine
        Ensure-Directory -Path $remoteTempDir
        
        # Copy requirements file to remote machine
        Copy-Item -Path $requirementsPath -Destination $remoteRequirementsPath -Force
        Write-Log "Copied requirements file to $Machine"

        # Check Python installation
        $pythonCheck = Invoke-Command -ComputerName $Machine -ScriptBlock {
            try {
                $ver = python --version 2>&1
                return $ver
            } catch {
                return $null
            }
        } -ErrorAction SilentlyContinue

        if ($null -eq $pythonCheck) {
            Write-Log "Installing Python on $Machine"
            
            # Download Python installer to the remote machine
            $remoteInstallerPath = "\\$Machine\$baseDir\python-3.11.0.exe"
            
            # Download the installer
            try {
                Invoke-WebRequest -Uri $pythonInstaller -OutFile $remoteInstallerPath -ErrorAction Stop
                Write-Log "Downloaded Python installer to $remoteInstallerPath"
            } catch {
                Write-Log "Failed to download Python installer: $($_.Exception.Message)"
                return $false
            }
            
            # Install Python on the remote machine
            $installResult = Invoke-Command -ComputerName $Machine -ScriptBlock {
                param($installerPath)
                try {
                    Start-Process -FilePath $installerPath -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python311' -Wait -NoNewWindow
                    return $true
                } catch {
                    return $false
                }
            } -ArgumentList $remoteInstallerPath -ErrorAction Continue
            
            if (-not $installResult) {
                Write-Log "Failed to install Python on $Machine"
                return $false
            }
            
            # Verify installation
            $pythonVerify = Invoke-Command -ComputerName $Machine -ScriptBlock {
                try {
                    # Refresh environment variables
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                    
                    # Check if Python is in PATH
                    $pythonPath = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
                    if ($pythonPath) {
                        $ver = python --version 2>&1
                        return "Python found at $pythonPath, version: $ver"
                    } else {
                        # Try explicit path
                        if (Test-Path "C:\Python311\python.exe") {
                            $ver = & "C:\Python311\python.exe" --version 2>&1
                            return "Python found at C:\Python311\python.exe, version: $ver"
                        } else {
                            return $null
                        }
                    }
                } catch {
                    return "Error checking Python: $($_.Exception.Message)"
                }
            } -ErrorAction SilentlyContinue
            
            if ($null -eq $pythonVerify) {
                Write-Log "Failed to verify Python on $Machine"
                return $false
            } else {
                Write-Log "Python verification on ${Machine}: $pythonVerify"
            }
        } else {
            Write-Log "Python already installed on ${Machine}: $pythonCheck"
        }

        # Create virtual environment on the remote machine in tempDir
        $venvSetup = Invoke-Command -ComputerName $Machine -ScriptBlock {
            param($localTempDir, $venvDirName, $requirementsPath)
            try {
                # Ensure temp directory exists locally
                if (!(Test-Path -Path $localTempDir)) {
                    New-Item -ItemType Directory -Path $localTempDir -Force | Out-Null
                }
                
                $venvPath = "$localTempDir\$venvDirName"
                
                # Determine Python executable
                if (Test-Path "C:\Python311\python.exe") {
                    $pythonExe = "C:\Python311\python.exe"
                } else {
                    $pythonExe = "python"
                }
                
                # Clean up existing venv if it exists
                if (Test-Path $venvPath) {
                    Remove-Item -Path $venvPath -Recurse -Force
                }
                
                # Create new venv
                & $pythonExe -m venv $venvPath
                
                if (-not (Test-Path "$venvPath\Scripts\activate.ps1")) {
                    return "Failed to create virtual environment at $venvPath"
                }
                
                # Activate and install requirements
                $activateScript = "$venvPath\Scripts\activate.ps1"
                & $activateScript
                & "$venvPath\Scripts\pip.exe" install --upgrade pip
                & "$venvPath\Scripts\pip.exe" install -r $requirementsPath
                
                # Test import
                $testImport = & "$venvPath\Scripts\python.exe" -c "import pandas; print('Pandas version:', pandas.__version__)"
                
                return "Virtual environment created at $venvPath. Test import: $testImport"
            } catch {
                return "Error setting up virtual environment: $($_.Exception.Message)"
            }
        } -ArgumentList $localTempDir, $venvDirName, $remoteRequirementsPath
        
        Write-Log "Virtual environment setup on ${Machine}: $venvSetup"
        
        return $true
    }
    catch {
        Write-Log "Error setting up Python on ${Machine}: $($_.ToString())"
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
            # Create machine-specific path
            $machineDir = "\\$machine\$baseDir\$tempDirName"
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

    # Copy script and config to each machine
    foreach ($machine in $Machines) {
        $machineDir = "\\$machine\$baseDir\$tempDirName"
        $machineScriptPath = "$machineDir\sample.py"
        $machineConfigPath = "$machineDir\config.ini"
        
        # Copy files
        Copy-Item -Path $ScriptPath -Destination $machineScriptPath -Force
        Copy-Item -Path $ConfigPath -Destination $machineConfigPath -Force
        Write-Log "Copied script and config to $machine"
    }

    $jobs = @()
    foreach ($machine in $Machines) {
        # Create local paths that will be used on the remote machine
        $localTempDir = "D:\RISHIN\framework_for_uni_py\SETUP_WORK\$tempDirName"
        $venvPath = "$localTempDir\$venvDirName"
        $localScriptPath = "$localTempDir\sample.py"
        $localConfigPath = "$localTempDir\config.ini"
        $localCsvPath = "$localTempDir\split_data_$machine.csv"
        
        $jobs += Start-Job -ScriptBlock {
            param($machine, $venvPath, $localScriptPath, $localConfigPath, $localCsvPath)
            try {
                $result = Invoke-Command -ComputerName $machine -ScriptBlock { 
                    param($venvPath, $scriptPath, $configPath, $csvPath)
                    try {
                        # Run with explicit paths
                        & "$venvPath\Scripts\activate.ps1"
                        & "$venvPath\Scripts\python.exe" $scriptPath --input $csvPath --config $configPath
                        return "Script executed successfully on $env:COMPUTERNAME"
                    } catch {
                        return "Error executing script: $($_.Exception.Message)"
                    }
                } -ArgumentList $venvPath, $localScriptPath, $localConfigPath, $localCsvPath
                
                return "Result from {$machine}: $result"
            }
            catch {
                return "Error executing script on ${machine}: $($_.ToString())"
            }
        } -ArgumentList $machine, $venvPath, $localScriptPath, $localConfigPath, $localCsvPath
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
    
    # Install Python and setup environments
    Write-Log "Setting up Python environments on target machines..."
    $pythonSetupResults = @()
    foreach ($machine in $machines) {
        Write-Log "Starting Python setup on $machine"
        $result = Install-PythonAndSetup -Machine $machine
        $pythonSetupResults += $result
        Write-Log "Python setup on $machine completed with result: $result"
    }

    if ($pythonSetupResults -contains $false) {
        Write-Log "WARNING: Python setup failed on one or more machines, proceeding anyway"
    }

    # Distribute CSV rows
    Write-Log "Distributing CSV data to machines..."
    $csvDistributionResult = Distribute-CSVRows -Machines $machines -CsvPath $csvPath
    
    if (-not $csvDistributionResult) {
        throw "Failed to distribute CSV data"
    }

    # Execute script on machines
    Write-Log "Executing script on machines..."
    Execute-ScriptOnMachines -Machines $machines -ScriptPath $scriptPath -ConfigPath $configPath

    Write-Log "Distributed processing completed"
}
catch {
    Write-Log "Critical error in distributed processing: $($_.ToString())"
}
finally {
    # Cleanup jobs
    Write-Log "Cleaning up background jobs"
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
}