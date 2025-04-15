package repository

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

// FileRepository handles file operations
type FileRepository struct {
	baseDataPath string
	modelPath    string
}

// NewFileRepository creates a new FileRepository instance
func NewFileRepository(baseDataPath string, modelPath string) *FileRepository {
	// Create base directories if they don't exist
	if err := os.MkdirAll(baseDataPath, 0755); err != nil {
		panic(fmt.Sprintf("Failed to create data directory: %v", err))
	}

	if err := os.MkdirAll(modelPath, 0755); err != nil {
		panic(fmt.Sprintf("Failed to create model directory: %v", err))
	}

	return &FileRepository{
		baseDataPath: baseDataPath,
		modelPath:    modelPath,
	}
}

// GetDataFilePath returns the full path to a data file
func (r *FileRepository) GetDataFilePath(fileName string) string {
	return filepath.Join(r.baseDataPath, fileName)
}

// GetModelPath returns the path to the model directory
func (r *FileRepository) GetModelPath() string {
	return r.modelPath
}

// FileExists checks if a file exists at the given path
func (r *FileRepository) FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// RunPythonScript executes a Python script with the given arguments
func (r *FileRepository) RunPythonScript(scriptPath string, args ...string) (string, error) {
	cmd := exec.Command("python", append([]string{scriptPath}, args...)...)

	// Create pipes for both stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stdout pipe: %v", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		return "", fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Combine both outputs
	output := ""

	// Start the command
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("failed to start Python script: %v", err)
	}

	// Read stdout in a goroutine
	stdoutDone := make(chan bool)
	go func() {
		stdoutBytes, _ := io.ReadAll(stdout)
		output += string(stdoutBytes)
		stdoutDone <- true
	}()

	// Read stderr
	stderrBytes, _ := io.ReadAll(stderr)
	output += string(stderrBytes)

	// Wait for stdout to be read
	<-stdoutDone

	// Wait for the command to complete
	if err := cmd.Wait(); err != nil {
		return output, fmt.Errorf("Python script failed: %v\nOutput: %s", err, output)
	}

	return output, nil
}

// ReadDataFile reads a file from the data directory
func (r *FileRepository) ReadDataFile(fileName string) ([]byte, error) {
	filePath := r.GetDataFilePath(fileName)

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open data file: %v", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read data file: %v", err)
	}

	return data, nil
}
