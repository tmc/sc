/**
 * Export Manager for Statechart Visualization
 * 
 * Provides comprehensive export functionality with support for multiple formats
 * including XState, React, Vue, SCXML, JSON, YAML, and diagram formats.
 */

class ExportManager {
    constructor() {
        this.currentMachine = null;
        this.supportedFormats = {};
        this.isInitialized = false;
        
        this.initializeUI();
        this.loadSupportedFormats();
    }

    async initializeUI() {
        // Create export modal
        this.createExportModal();
        this.setupEventListeners();
    }

    createExportModal() {
        // Remove existing modal if present
        const existingModal = document.getElementById('export-modal');
        if (existingModal) {
            existingModal.remove();
        }

        const modal = document.createElement('div');
        modal.id = 'export-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-container">
                <div class="modal-header">
                    <h2>Export Statechart</h2>
                    <button class="modal-close" type="button">×</button>
                </div>
                
                <div class="modal-content">
                    <div class="export-sections">
                        <!-- Format Selection -->
                        <div class="export-section">
                            <h3>Format</h3>
                            <div class="format-grid" id="format-grid">
                                <div class="format-loading">Loading formats...</div>
                            </div>
                        </div>
                        
                        <!-- Export Options -->
                        <div class="export-section">
                            <h3>Options</h3>
                            <div class="options-container" id="export-options">
                                <div class="option-group">
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="option-pretty" checked>
                                        <span class="checkbox-mark"></span>
                                        Pretty format output
                                    </label>
                                </div>
                                
                                <div class="option-group">
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="option-typescript">
                                        <span class="checkbox-mark"></span>
                                        Generate TypeScript definitions
                                    </label>
                                </div>
                                
                                <div class="option-group">
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="option-comments" checked>
                                        <span class="checkbox-mark"></span>
                                        Include comments
                                    </label>
                                </div>
                                
                                <div class="option-group">
                                    <label for="machine-name">Machine Name:</label>
                                    <input type="text" id="machine-name" placeholder="Auto-generated">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Preview -->
                        <div class="export-section">
                            <h3>Preview</h3>
                            <div class="preview-container">
                                <div class="preview-header">
                                    <span class="preview-filename" id="preview-filename">output.js</span>
                                    <button class="preview-copy" id="copy-preview" title="Copy to clipboard">
                                        <svg viewBox="0 0 24 24">
                                            <path d="M19,21H8V7H19M19,5H8A2,2 0 0,0 6,7V21A2,2 0 0,0 8,23H19A2,2 0 0,0 21,21V7A2,2 0 0,0 19,5M16,1H4A2,2 0 0,0 2,3V17H4V3H16V1Z"/>
                                        </svg>
                                    </button>
                                </div>
                                <pre class="preview-content" id="preview-content">Select a format to see preview...</pre>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="modal-footer">
                    <div class="export-info">
                        <span class="export-status" id="export-status">Ready to export</span>
                    </div>
                    <div class="modal-actions">
                        <button class="btn-secondary" id="cancel-export">Cancel</button>
                        <button class="btn-primary" id="download-export" disabled>Download</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        this.modal = modal;
    }

    setupEventListeners() {
        // Modal controls
        this.modal.querySelector('.modal-close').addEventListener('click', () => this.hideModal());
        this.modal.querySelector('#cancel-export').addEventListener('click', () => this.hideModal());
        this.modal.querySelector('#download-export').addEventListener('click', () => this.downloadExport());
        this.modal.querySelector('#copy-preview').addEventListener('click', () => this.copyPreview());

        // Close modal on overlay click
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hideModal();
            }
        });

        // Option changes
        const options = this.modal.querySelectorAll('#export-options input');
        options.forEach(option => {
            option.addEventListener('change', () => this.updatePreview());
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.classList.contains('active')) {
                this.hideModal();
            }
        });

        // Wire up main export button
        const exportButton = document.querySelector('[title="Export"]');
        if (exportButton) {
            exportButton.addEventListener('click', () => this.showExportModal());
        }
    }

    async loadSupportedFormats() {
        try {
            const response = await fetch('/api/v1/export/formats');
            const data = await response.json();
            this.supportedFormats = data.formats;
            this.renderFormatGrid();
            this.isInitialized = true;
        } catch (error) {
            console.error('Failed to load supported formats:', error);
            this.showError('Failed to load export formats');
        }
    }

    renderFormatGrid() {
        const grid = this.modal.querySelector('#format-grid');
        grid.innerHTML = '';

        Object.entries(this.supportedFormats).forEach(([key, format]) => {
            const formatCard = document.createElement('div');
            formatCard.className = 'format-card';
            formatCard.dataset.format = key;
            
            formatCard.innerHTML = `
                <div class="format-header">
                    <h4>${format.name}</h4>
                    <div class="format-extensions">
                        ${format.extensions.map(ext => `<span class="ext">.${ext}</span>`).join('')}
                    </div>
                </div>
                <p class="format-description">${format.description}</p>
                <div class="format-features">
                    ${format.features.slice(0, 3).map(feature => `<span class="feature-tag">${feature}</span>`).join('')}
                </div>
            `;

            formatCard.addEventListener('click', () => this.selectFormat(key, format));
            grid.appendChild(formatCard);
        });
    }

    selectFormat(formatKey, format) {
        // Update UI selection
        this.modal.querySelectorAll('.format-card').forEach(card => {
            card.classList.remove('selected');
        });
        this.modal.querySelector(`[data-format="${formatKey}"]`).classList.add('selected');

        this.selectedFormat = formatKey;
        this.selectedFormatInfo = format;

        // Update options based on format
        this.updateOptionsForFormat(formatKey);
        
        // Update preview
        this.updatePreview();

        // Enable download button
        this.modal.querySelector('#download-export').disabled = false;
    }

    updateOptionsForFormat(formatKey) {
        const tsOption = this.modal.querySelector('#option-typescript');
        const tsGroup = tsOption.closest('.option-group');
        
        // Show TypeScript option for supported formats
        const supportsTypeScript = ['xstate', 'react', 'vue'].includes(formatKey);
        tsGroup.style.display = supportsTypeScript ? 'block' : 'none';
        
        // Auto-check TypeScript for TypeScript formats
        if (formatKey.includes('ts') || formatKey.includes('tsx')) {
            tsOption.checked = true;
            tsOption.disabled = true;
        } else {
            tsOption.disabled = false;
        }

        // Set default machine name
        const machineNameInput = this.modal.querySelector('#machine-name');
        if (this.currentMachine && !machineNameInput.value) {
            machineNameInput.value = this.currentMachine.id;
        }
    }

    async updatePreview() {
        if (!this.selectedFormat || !this.currentMachine) return;

        const options = this.gatherOptions();
        const filenameElement = this.modal.querySelector('#preview-filename');
        const contentElement = this.modal.querySelector('#preview-content');
        
        // Update filename
        const extension = this.selectedFormatInfo.extensions[0];
        const machineName = options.machineName || this.currentMachine.id;
        filenameElement.textContent = `${machineName}.${extension}`;

        // Show loading
        contentElement.textContent = 'Generating preview...';
        contentElement.className = 'preview-content loading';

        try {
            const response = await fetch(`/api/v1/machines/${this.currentMachine.id}/export/${this.selectedFormat}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    format: this.selectedFormat, 
                    options: { ...options, preview: true }
                }),
            });

            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                    contentElement.textContent = result.content;
                    contentElement.className = 'preview-content';
                    this.updateStatus(`Preview generated • ${result.metadata?.stateCount || 0} states • ${result.metadata?.transitionCount || 0} transitions`);
                } else {
                    contentElement.textContent = `Error: ${result.error}`;
                    contentElement.className = 'preview-content error';
                    this.updateStatus(`Preview error: ${result.error}`);
                }
            } else {
                const error = await response.text();
                contentElement.textContent = `HTTP Error: ${error}`;
                contentElement.className = 'preview-content error';
                this.updateStatus(`Preview failed: ${error}`);
            }
        } catch (error) {
            contentElement.textContent = `Network Error: ${error.message}`;
            contentElement.className = 'preview-content error';
            this.updateStatus(`Preview failed: ${error.message}`);
        }
    }

    gatherOptions() {
        const options = {};
        
        // Boolean options
        options.pretty = this.modal.querySelector('#option-pretty').checked;
        options.typescript = this.modal.querySelector('#option-typescript').checked;
        options.comments = this.modal.querySelector('#option-comments').checked;
        
        // String options
        const machineName = this.modal.querySelector('#machine-name').value.trim();
        if (machineName) {
            options.machineName = machineName;
        }

        return options;
    }

    async downloadExport() {
        if (!this.selectedFormat || !this.currentMachine) return;

        const downloadButton = this.modal.querySelector('#download-export');
        const originalText = downloadButton.textContent;
        
        downloadButton.disabled = true;
        downloadButton.textContent = 'Downloading...';
        this.updateStatus('Preparing download...');

        try {
            const options = this.gatherOptions();
            const url = `/api/v1/machines/${this.currentMachine.id}/export/${this.selectedFormat}?download=true`;
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    format: this.selectedFormat, 
                    options: options
                }),
            });

            if (response.ok) {
                // Get filename from Content-Disposition header or generate
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `${this.currentMachine.id}.${this.selectedFormatInfo.extensions[0]}`;
                
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                    if (filenameMatch) {
                        filename = filenameMatch[1];
                    }
                }

                // Create download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);

                this.updateStatus(`Downloaded ${filename}`);
                setTimeout(() => this.hideModal(), 1000);
            } else {
                const error = await response.text();
                this.showError(`Download failed: ${error}`);
            }
        } catch (error) {
            this.showError(`Download error: ${error.message}`);
        } finally {
            downloadButton.disabled = false;
            downloadButton.textContent = originalText;
        }
    }

    async copyPreview() {
        const content = this.modal.querySelector('#preview-content').textContent;
        const copyButton = this.modal.querySelector('#copy-preview');
        
        try {
            await navigator.clipboard.writeText(content);
            
            // Visual feedback
            const originalHTML = copyButton.innerHTML;
            copyButton.innerHTML = `
                <svg viewBox="0 0 24 24">
                    <path d="M21,7L9,19L3.5,13.5L4.91,12.09L9,16.17L19.59,5.59L21,7Z"/>
                </svg>
            `;
            copyButton.classList.add('success');
            
            setTimeout(() => {
                copyButton.innerHTML = originalHTML;
                copyButton.classList.remove('success');
            }, 1000);
            
            this.updateStatus('Code copied to clipboard');
        } catch (error) {
            this.showError('Failed to copy to clipboard');
        }
    }

    showExportModal() {
        if (!this.currentMachine) {
            this.showError('No machine loaded to export');
            return;
        }

        if (!this.isInitialized) {
            this.showError('Export system not ready');
            return;
        }

        this.modal.classList.add('active');
        this.updateStatus('Select a format to export');
        
        // Reset form
        this.resetForm();
    }

    hideModal() {
        this.modal.classList.remove('active');
    }

    resetForm() {
        // Reset format selection
        this.modal.querySelectorAll('.format-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Reset options
        this.modal.querySelector('#option-pretty').checked = true;
        this.modal.querySelector('#option-typescript').checked = false;
        this.modal.querySelector('#option-comments').checked = true;
        this.modal.querySelector('#machine-name').value = this.currentMachine?.id || '';
        
        // Reset preview
        this.modal.querySelector('#preview-content').textContent = 'Select a format to see preview...';
        this.modal.querySelector('#preview-content').className = 'preview-content';
        this.modal.querySelector('#preview-filename').textContent = 'output.js';
        
        // Reset buttons
        this.modal.querySelector('#download-export').disabled = true;
        
        this.selectedFormat = null;
        this.selectedFormatInfo = null;
    }

    setMachine(machine) {
        this.currentMachine = machine;
        
        // Update machine name in options if modal is open
        if (this.modal.classList.contains('active')) {
            const machineNameInput = this.modal.querySelector('#machine-name');
            if (!machineNameInput.value) {
                machineNameInput.value = machine.id;
            }
        }
    }

    updateStatus(message) {
        const statusElement = this.modal.querySelector('#export-status');
        statusElement.textContent = message;
        statusElement.className = 'export-status';
    }

    showError(message) {
        const statusElement = this.modal.querySelector('#export-status');
        statusElement.textContent = message;
        statusElement.className = 'export-status error';
        
        // Also show as alert for important errors
        if (message.includes('Failed') || message.includes('Error')) {
            console.error('Export Error:', message);
        }
    }

    // Public API for integration
    exportMachine(machine, format = null) {
        this.setMachine(machine);
        
        if (format) {
            this.selectedFormat = format;
            this.selectedFormatInfo = this.supportedFormats[format];
        }
        
        this.showExportModal();
    }

    getSupportedFormats() {
        return this.supportedFormats;
    }

    isReady() {
        return this.isInitialized;
    }
}

// Export for global access
if (typeof window !== 'undefined') {
    window.ExportManager = ExportManager;
}