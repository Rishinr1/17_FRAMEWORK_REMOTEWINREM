# Define parameters
$machines = Get-Content "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\machine_list.txt"  # List of target machines
$pythonInstaller = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
$baseDir = "D$\RISHIN\framework_for_uni_py\SETUP_WORK"  # Base directory on each machine
$tempDirName = "temp" # Temporary directory name
$scriptPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\sample_backup_4.py"
$configPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\config.ini"
$requirementsPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\requirements.txt"
$logPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\setup_log.txt"
$venvDirName="venv"
$folderBPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\entirefolderB"  # Replace with actual path to folderB
$folderCPath = "\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\SETUP_WORK\folderC"  # Replace with actual path to folderC
# Destination path on each machine (will be created if doesn't exist)
$destinationPath = "D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp\copied_path"
$destinationPath2 = "D$\RISHIN\framework_for_uni_py\SETUP_WORK\temp\copied_path_inp"

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

function Ensure-RemoteDirectory {
    param(
        [string]$Machine,
        [string]$RemotePath
    )

    try {
        Write-Log "Ensuring directory exists on $Machine`: $RemotePath"
        Invoke-Command -ComputerName $Machine -ScriptBlock {
            param($path)
            if (!(Test-Path -Path $path)) {
                New-Item -ItemType Directory -Path $path -Force | Out-Null
                Write-Output "Created directory $path"
            } else {
                Write-Output "Directory already exists $path"
            }
        } -ArgumentList $RemotePath -ErrorAction Stop | Out-String | Write-Log
    } catch {
        Write-Log "Error creating directory on $Machine`: $($_.Exception.Message)"
        throw
    }
}

function Copy-FileAndFolderB {
    param(
        [string[]]$Machines,
        [string]$FolderBPath,
        [string]$DestinationPath
    )

    foreach ($machine in $Machines) {
        try {
            # Full UNC path to destination
            $remoteMachinePath = "\\$machine\$DestinationPath"
            
            # Ensure remote directory exists
            Ensure-RemoteDirectory -Machine $machine -RemotePath $remoteMachinePath

            $remoteFolderB = "$remoteMachinePath\entirefolderB"
       
            Write-Log "Copying $FolderBPath to $remoteFolderB"
            if (Test-Path -Path $FolderBPath) {
                # Create target directory first
                if (!(Test-Path -Path $remoteFolderB)) {
                    New-Item -ItemType Directory -Path $remoteFolderB -Force | Out-Null
                }
                
                # Use robocopy for better handling of folder copying
                $robocopyResult = robocopy $FolderBPath $remoteFolderB /E /R:3 /W:5 /MT:8 /NFL /NDL
                Write-Log "Robocopy completed with exit code: $LASTEXITCODE"
            } else {
                Write-Log "Warning: Source folder $FolderBPath does not exist"
            }
            
            Write-Log "Completed copying to $machine at $remoteMachinePath"
        } catch {
            Write-Log "Error copying to $machine`: $($_.Exception.Message)"
        }
    }
}

function Distribute-FolderCSubfolders {
    param(
        [string[]]$Machines,
        [string]$FolderCPath,
        [string]$DestinationPath2
    )

    # Verify source folder exists
    if (!(Test-Path -Path $FolderCPath)) {
        Write-Log "Error: Source folder $FolderCPath does not exist"
        return
    }

    # Get all final-level subfolders inside FolderC
    $subfolders = Get-ChildItem -Path $FolderCPath -Recurse -Directory | Where-Object { 
        -not ($_ | Get-ChildItem -Directory)
    }
    
    $totalFolders = $subfolders.Count
    $foldersPerMachine = [math]::Ceiling($totalFolders / $Machines.Count)
    $index = 0

    Write-Log "Distributing $totalFolders subfolders from FolderC across $($Machines.Count) machines"

    foreach ($machine in $Machines) {
        $machineFolders = $subfolders | Select-Object -Skip $index -First $foldersPerMachine
        $index += $foldersPerMachine

        if ($machineFolders.Count -eq 0) { 
            Write-Log "No folders to copy to $machine"
            continue 
        }

        # Create full UNC path to destination
        $remoteDestPath = "\\$machine\$DestinationPath2"
        
        Write-Log "Ensuring remote directory exists on $machine..."
        try {
            Ensure-RemoteDirectory -Machine $machine -RemotePath $remoteDestPath
            
            foreach ($folder in $machineFolders) {
                $folderName = Split-Path $folder -Leaf
                $remoteFolder = "$remoteDestPath\$folderName"
                $sourcePath = $folder.FullName
                
                Write-Log "Copying folder $folderName from $sourcePath to $machine"
                
                # Create destination folder
                if (!(Test-Path -Path $remoteFolder)) {
                    New-Item -ItemType Directory -Path $remoteFolder -Force | Out-Null
                }
                
                # Use robocopy for reliable copying
                $robocopyArgs = @(
                    $sourcePath,
                    $remoteFolder,
                    "/E",         # Copy subdirectories, including empty ones
                    "/R:3",       # Retry 3 times
                    "/W:5",       # Wait 5 seconds between retries
                    "/MT:8",      # 8 threads
                    "/NFL",       # No file list
                    "/NDL"        # No directory list
                )
                
                $robocopyProcess = Start-Process -FilePath "robocopy" -ArgumentList $robocopyArgs -NoNewWindow -Wait -PassThru
                
                # Process robocopy exit codes (0-7 are considered success)
                if ($robocopyProcess.ExitCode -lt 8) {
                    Write-Log "Successfully copied folder $folderName to $machine"
                } else {
                    Write-Log "Error copying folder $folderName to $machine. Exit code: $($robocopyProcess.ExitCode)"
                }
            }
        } catch {
            Write-Log "Error processing machine $machine`: $($_.Exception.Message)"
        }
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


# Execute script on machines
function Execute-ScriptOnMachines {
    param(
        [string[]]$Machines,
        [string]$ScriptPath,
        [string]$ConfigPath
    )

    # Copy script and config to each machine
    foreach ($machine in $Machines) {
        try {
            $machineDir = "\\$machine\$baseDir\$tempDirName"
            $machineScriptPath = "$machineDir\sample.py"
            $machineConfigPath = "$machineDir\config.ini"
            
            # Ensure directory exists
            Ensure-Directory -Path $machineDir
            
            # Copy files
            Copy-Item -Path $ScriptPath -Destination $machineScriptPath -Force
            Copy-Item -Path $ConfigPath -Destination $machineConfigPath -Force
            Write-Log "Copied script and config to $machine"
        } catch {
            Write-Log "Error copying script to {$machine}: $($_.Exception.Message)"
        }
    }

    $jobs = @()
    foreach ($machine in $Machines) {
        # Create local paths that will be used on the remote machine
        $localTempDir = "D:\RISHIN\framework_for_uni_py\SETUP_WORK\$tempDirName"
        $venvPath = "$localTempDir\$venvDirName"
        $localScriptPath = "$localTempDir\sample.py"
        $localConfigPath = "$localTempDir\config.ini"
        
        $jobs += Start-Job -ScriptBlock {
            param($machine, $venvPath, $localScriptPath, $localConfigPath)
            try {
                $result = Invoke-Command -ComputerName $machine -ScriptBlock { 
                    param($venvPath, $scriptPath, $configPath)
                    try {
                        # Verify paths exist before execution
                        if (!(Test-Path $venvPath)) {
                            return "ERROR: Virtual environment not found at $venvPath"
                        }
                        if (!(Test-Path $scriptPath)) {
                            return "ERROR: Script not found at $scriptPath"
                        }
                        if (!(Test-Path $configPath)) {
                            return "ERROR: Config not found at $configPath"
                        }
                        
                        # Run with explicit paths
                        & "$venvPath\Scripts\activate.ps1"
                        & "$venvPath\Scripts\python.exe" $scriptPath --config $configPath
                        return "Script executed successfully on $env:COMPUTERNAME"
                    } catch {
                        return "Error executing script: $($_.Exception.Message)"
                    }
                } -ArgumentList $venvPath, $localScriptPath, $localConfigPath
                
                return "Result from $machine`: $result"
            }
            catch {
                return "Error executing script on $machine`: $($_.ToString())"
            }
        } -ArgumentList $machine, $venvPath, $localScriptPath, $localConfigPath
    }

    # Wait and collect results
    $results = $jobs | Wait-Job | Receive-Job
    $results | ForEach-Object { Write-Log $_ }
}

# Main Execution Flow  pat
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

    

    # Copy FileA and FolderB to each machine
    Write-Log "Copying FileA and FolderB to destination path on machines..."
    Copy-FileAndFolderB -Machines $machines -FolderBPath $folderBPath -DestinationPath $destinationPath

    # Distribute FolderC subfolders across machines
    Write-Log "Distributing FolderC subfolders across machines..."
    Distribute-FolderCSubfolders -Machines $machines -FolderCPath $folderCPath -DestinationPath2 $destinationPath2

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