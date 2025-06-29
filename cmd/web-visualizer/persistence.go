package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/tmc/sc/semantics/v1"
)

// MachineSnapshot represents a persisted machine state
type MachineSnapshot struct {
	ID            string                 `json:"id"`
	State         string                 `json:"state"`
	Configuration []string               `json:"configuration"`
	Context       map[string]interface{} `json:"context"`
	Statechart    interface{}            `json:"statechart"`
	StepHistory   interface{}            `json:"step_history"`
	Timestamp     time.Time              `json:"timestamp"`
	Version       int                    `json:"version"`
}

// SaveMachine persists a machine to disk
func (pm *PersistenceManager) SaveMachine(id string, machine *semantics.MachineWrapper) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	snapshot := MachineSnapshot{
		ID:            machine.Id,
		State:         machine.State.String(),
		Configuration: getStateLabels(machine.Configuration.States),
		Context:       machine.Context.AsMap(),
		Statechart:    machine.Statechart,
		StepHistory:   machine.StepHistory,
		Timestamp:     time.Now(),
		Version:       1,
	}

	filename := filepath.Join(pm.dataDir, fmt.Sprintf("%s.json", id))
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create machine file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(snapshot); err != nil {
		return fmt.Errorf("failed to encode machine snapshot: %w", err)
	}

	return nil
}

// LoadMachine loads a machine from disk
func (pm *PersistenceManager) LoadMachine(id string) (*MachineSnapshot, error) {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	filename := filepath.Join(pm.dataDir, fmt.Sprintf("%s.json", id))
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open machine file: %w", err)
	}
	defer file.Close()

	var snapshot MachineSnapshot
	if err := json.NewDecoder(file).Decode(&snapshot); err != nil {
		return nil, fmt.Errorf("failed to decode machine snapshot: %w", err)
	}

	return &snapshot, nil
}

// LoadMachines loads all persisted machines
func (pm *PersistenceManager) LoadMachines(machines map[string]*semantics.MachineWrapper) error {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	files, err := filepath.Glob(filepath.Join(pm.dataDir, "*.json"))
	if err != nil {
		return fmt.Errorf("failed to list machine files: %w", err)
	}

	for _, filename := range files {
		file, err := os.Open(filename)
		if err != nil {
			continue // Skip files that can't be opened
		}

		var snapshot MachineSnapshot
		if err := json.NewDecoder(file).Decode(&snapshot); err != nil {
			file.Close()
			continue // Skip files that can't be decoded
		}
		file.Close()

		// Try to recreate the machine from the snapshot
		// This is a simplified approach - in production you'd want more robust restoration
		// For now, we'll just log that we found persisted machines
		fmt.Printf("Found persisted machine: %s (saved at %v)\n", snapshot.ID, snapshot.Timestamp)
	}

	return nil
}

// DeleteMachine removes a machine from persistence
func (pm *PersistenceManager) DeleteMachine(id string) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	filename := filepath.Join(pm.dataDir, fmt.Sprintf("%s.json", id))
	if err := os.Remove(filename); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to delete machine file: %w", err)
	}

	return nil
}

// ListMachines returns a list of all persisted machine IDs
func (pm *PersistenceManager) ListMachines() ([]string, error) {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	files, err := filepath.Glob(filepath.Join(pm.dataDir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to list machine files: %w", err)
	}

	var ids []string
	for _, filename := range files {
		base := filepath.Base(filename)
		id := base[:len(base)-5] // Remove .json extension
		ids = append(ids, id)
	}

	return ids, nil
}

// CreateBackup creates a backup of all persisted machines
func (pm *PersistenceManager) CreateBackup() error {
	pm.mutex.RLock()
	defer pm.mutex.RUnlock()

	backupDir := filepath.Join(pm.dataDir, "backups", time.Now().Format("2006-01-02_15-04-05"))
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	files, err := filepath.Glob(filepath.Join(pm.dataDir, "*.json"))
	if err != nil {
		return fmt.Errorf("failed to list machine files: %w", err)
	}

	for _, filename := range files {
		src, err := os.Open(filename)
		if err != nil {
			continue
		}

		base := filepath.Base(filename)
		dst, err := os.Create(filepath.Join(backupDir, base))
		if err != nil {
			src.Close()
			continue
		}

		// Copy file contents
		_, err = dst.ReadFrom(src)
		src.Close()
		dst.Close()

		if err != nil {
			return fmt.Errorf("failed to copy machine file: %w", err)
		}
	}

	return nil
}

// RestoreBackup restores machines from a backup
func (pm *PersistenceManager) RestoreBackup(backupPath string) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	files, err := filepath.Glob(filepath.Join(backupPath, "*.json"))
	if err != nil {
		return fmt.Errorf("failed to list backup files: %w", err)
	}

	for _, filename := range files {
		src, err := os.Open(filename)
		if err != nil {
			continue
		}

		base := filepath.Base(filename)
		dst, err := os.Create(filepath.Join(pm.dataDir, base))
		if err != nil {
			src.Close()
			continue
		}

		// Copy file contents
		_, err = dst.ReadFrom(src)
		src.Close()
		dst.Close()

		if err != nil {
			return fmt.Errorf("failed to restore machine file: %w", err)
		}
	}

	return nil
}

// GetMachineHistory returns the version history of a machine (placeholder)
func (pm *PersistenceManager) GetMachineHistory(id string) ([]MachineSnapshot, error) {
	// In a real implementation, this would maintain version history
	// For now, just return the current snapshot
	snapshot, err := pm.LoadMachine(id)
	if err != nil {
		return nil, err
	}

	return []MachineSnapshot{*snapshot}, nil
}

// CleanupOldBackups removes backups older than the specified duration
func (pm *PersistenceManager) CleanupOldBackups(maxAge time.Duration) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	backupDir := filepath.Join(pm.dataDir, "backups")
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		return nil // No backups directory
	}

	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return fmt.Errorf("failed to read backup directory: %w", err)
	}

	cutoff := time.Now().Add(-maxAge)
	for _, entry := range entries {
		if entry.IsDir() {
			info, err := entry.Info()
			if err != nil {
				continue
			}

			if info.ModTime().Before(cutoff) {
				backupPath := filepath.Join(backupDir, entry.Name())
				if err := os.RemoveAll(backupPath); err != nil {
					fmt.Printf("Failed to remove old backup %s: %v\n", entry.Name(), err)
				}
			}
		}
	}

	return nil
}