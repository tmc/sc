/**
 * AIGenerator - Modal interface for AI-assisted statechart generation
 *
 * Features:
 * - Natural language description input
 * - Domain selector for context hints
 * - Example scenarios input
 * - Streaming preview of generated statechart
 * - Apply/Cancel buttons
 * - Improve and explain existing statecharts
 */

class AIGenerator extends EventTarget {
    constructor(options = {}) {
        super();

        this.apiBaseUrl = options.apiBaseUrl || '/api/v1';
        this.stateManager = options.stateManager;

        // State
        this.isOpen = false;
        this.isGenerating = false;
        this.generatedStatechart = null;
        this.mode = 'generate'; // 'generate' | 'improve' | 'explain'
        this.currentStatechart = null;

        // Domain options
        this.domains = [
            { value: 'general', label: 'General' },
            { value: 'ui', label: 'User Interface' },
            { value: 'game', label: 'Game Logic' },
            { value: 'workflow', label: 'Business Workflow' },
            { value: 'device', label: 'Device/IoT' },
            { value: 'auth', label: 'Authentication' }
        ];

        this.createModal();
        this.setupEventListeners();
        this.checkAIStatus();
    }

    createModal() {
        this.modal = document.createElement('div');
        this.modal.className = 'ai-generator-modal';
        this.modal.innerHTML = `
            <div class="ai-generator-backdrop"></div>
            <div class="ai-generator-dialog">
                <div class="ai-generator-header">
                    <h2 class="ai-generator-title">Generate Statechart with AI</h2>
                    <button class="ai-generator-close" aria-label="Close">×</button>
                </div>

                <div class="ai-generator-body">
                    <div class="ai-generator-form">
                        <div class="ai-form-group">
                            <label for="ai-description">Description</label>
                            <textarea
                                id="ai-description"
                                placeholder="Describe the statechart behavior you want to create...&#10;&#10;Example: A traffic light that cycles through red, yellow, and green states with timed transitions."
                                rows="4"
                            ></textarea>
                        </div>

                        <div class="ai-form-group">
                            <label for="ai-domain">Domain</label>
                            <select id="ai-domain">
                                ${this.domains.map(d =>
                                    `<option value="${d.value}">${d.label}</option>`
                                ).join('')}
                            </select>
                        </div>

                        <div class="ai-form-group">
                            <label for="ai-scenarios">Example Scenarios (optional)</label>
                            <textarea
                                id="ai-scenarios"
                                placeholder="List scenarios the statechart should handle...&#10;- User clicks the button&#10;- Timer expires&#10;- Error occurs"
                                rows="3"
                            ></textarea>
                        </div>
                    </div>

                    <div class="ai-generator-preview">
                        <div class="ai-preview-header">
                            <span>Preview</span>
                            <span class="ai-preview-status"></span>
                        </div>
                        <div class="ai-preview-content">
                            <pre class="ai-preview-json"></pre>
                        </div>
                    </div>
                </div>

                <div class="ai-generator-footer">
                    <div class="ai-status">
                        <span class="ai-status-indicator"></span>
                        <span class="ai-status-text">Checking AI availability...</span>
                    </div>
                    <div class="ai-actions">
                        <button class="ai-btn ai-btn-secondary ai-btn-cancel">Cancel</button>
                        <button class="ai-btn ai-btn-primary ai-btn-generate" disabled>Generate</button>
                        <button class="ai-btn ai-btn-primary ai-btn-apply" style="display: none;">Apply</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(this.modal);

        // Cache DOM references
        this.backdrop = this.modal.querySelector('.ai-generator-backdrop');
        this.dialog = this.modal.querySelector('.ai-generator-dialog');
        this.closeBtn = this.modal.querySelector('.ai-generator-close');
        this.cancelBtn = this.modal.querySelector('.ai-btn-cancel');
        this.generateBtn = this.modal.querySelector('.ai-btn-generate');
        this.applyBtn = this.modal.querySelector('.ai-btn-apply');
        this.descriptionInput = this.modal.querySelector('#ai-description');
        this.domainSelect = this.modal.querySelector('#ai-domain');
        this.scenariosInput = this.modal.querySelector('#ai-scenarios');
        this.previewContent = this.modal.querySelector('.ai-preview-json');
        this.previewStatus = this.modal.querySelector('.ai-preview-status');
        this.statusIndicator = this.modal.querySelector('.ai-status-indicator');
        this.statusText = this.modal.querySelector('.ai-status-text');
        this.titleEl = this.modal.querySelector('.ai-generator-title');
    }

    setupEventListeners() {
        this.backdrop.addEventListener('click', () => this.close());
        this.closeBtn.addEventListener('click', () => this.close());
        this.cancelBtn.addEventListener('click', () => this.close());
        this.generateBtn.addEventListener('click', () => this.generate());
        this.applyBtn.addEventListener('click', () => this.apply());

        // Enable generate button when description is entered
        this.descriptionInput.addEventListener('input', () => {
            this.generateBtn.disabled = !this.descriptionInput.value.trim() || !this.aiAvailable;
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!this.isOpen) return;

            if (e.key === 'Escape') {
                this.close();
            } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                if (!this.generateBtn.disabled) {
                    this.generate();
                }
            }
        });
    }

    async checkAIStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/ai/status`);
            const data = await response.json();

            this.aiAvailable = data.available;

            if (data.available) {
                this.statusIndicator.classList.add('available');
                this.statusText.textContent = `AI Ready (${data.provider})`;
                this.generateBtn.disabled = !this.descriptionInput.value.trim();
            } else {
                this.statusIndicator.classList.add('unavailable');
                this.statusText.textContent = data.error || 'AI not available';
                this.generateBtn.disabled = true;
            }
        } catch (error) {
            this.aiAvailable = false;
            this.statusIndicator.classList.add('unavailable');
            this.statusText.textContent = 'Failed to check AI status';
            this.generateBtn.disabled = true;
        }
    }

    open(mode = 'generate', statechart = null) {
        this.mode = mode;
        this.currentStatechart = statechart;
        this.generatedStatechart = null;

        // Update UI for mode
        this.updateModeUI();

        // Show modal
        this.modal.classList.add('open');
        this.isOpen = true;

        // Focus description input
        setTimeout(() => this.descriptionInput.focus(), 100);
    }

    updateModeUI() {
        switch (this.mode) {
            case 'generate':
                this.titleEl.textContent = 'Generate Statechart with AI';
                this.generateBtn.textContent = 'Generate';
                this.descriptionInput.placeholder = 'Describe the statechart behavior you want to create...';
                break;
            case 'improve':
                this.titleEl.textContent = 'Improve Statechart with AI';
                this.generateBtn.textContent = 'Improve';
                this.descriptionInput.placeholder = 'Describe what improvements you want...';
                if (this.currentStatechart) {
                    this.previewContent.textContent = JSON.stringify(this.currentStatechart, null, 2);
                }
                break;
            case 'explain':
                this.titleEl.textContent = 'Explain Statechart';
                this.generateBtn.textContent = 'Explain';
                this.descriptionInput.placeholder = 'What do you want to know about this statechart?';
                if (this.currentStatechart) {
                    this.previewContent.textContent = JSON.stringify(this.currentStatechart, null, 2);
                }
                break;
        }

        this.applyBtn.style.display = 'none';
        this.generateBtn.style.display = '';
        this.previewStatus.textContent = '';
    }

    close() {
        this.modal.classList.remove('open');
        this.isOpen = false;
        this.isGenerating = false;
        this.generatedStatechart = null;

        // Clear form
        this.descriptionInput.value = '';
        this.scenariosInput.value = '';
        this.domainSelect.value = 'general';
        this.previewContent.textContent = '';
        this.previewStatus.textContent = '';
    }

    async generate() {
        if (this.isGenerating) return;

        this.isGenerating = true;
        this.generateBtn.disabled = true;
        this.previewStatus.textContent = 'Generating...';
        this.previewContent.textContent = '';

        const description = this.descriptionInput.value.trim();
        const domain = this.domainSelect.value;
        const scenarios = this.scenariosInput.value
            .split('\n')
            .map(s => s.trim())
            .filter(s => s);

        try {
            let endpoint, body;

            switch (this.mode) {
                case 'generate':
                    endpoint = `${this.apiBaseUrl}/ai/generate`;
                    body = { description, domain, scenarios };
                    break;
                case 'improve':
                    endpoint = `${this.apiBaseUrl}/ai/improve`;
                    body = { statechart: this.currentStatechart, description };
                    break;
                case 'explain':
                    endpoint = `${this.apiBaseUrl}/ai/explain`;
                    body = { statechart: this.currentStatechart, description };
                    break;
            }

            // Try streaming first
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(body)
            });

            if (response.headers.get('content-type')?.includes('text/event-stream')) {
                await this.handleStreamingResponse(response);
            } else {
                // Non-streaming response
                const data = await response.json();

                if (this.mode === 'explain') {
                    this.previewContent.textContent = data.explanation;
                    this.previewStatus.textContent = 'Explanation ready';
                } else {
                    this.generatedStatechart = data;
                    this.previewContent.textContent = JSON.stringify(data, null, 2);
                    this.previewStatus.textContent = 'Generated successfully';

                    // Show apply button
                    this.generateBtn.style.display = 'none';
                    this.applyBtn.style.display = '';
                }
            }
        } catch (error) {
            console.error('AI generation error:', error);
            this.previewStatus.textContent = `Error: ${error.message}`;
            this.previewContent.textContent = '';
        } finally {
            this.isGenerating = false;
            this.generateBtn.disabled = false;
        }
    }

    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        // Unescape newlines
                        const content = data.replace(/\\n/g, '\n');
                        accumulated += content;
                        this.previewContent.textContent = accumulated;
                        this.previewContent.scrollTop = this.previewContent.scrollHeight;
                    } else if (line.startsWith('event: done')) {
                        this.previewStatus.textContent = 'Generated successfully';

                        // Try to parse the accumulated content as JSON
                        try {
                            this.generatedStatechart = JSON.parse(accumulated);
                            this.previewContent.textContent = JSON.stringify(this.generatedStatechart, null, 2);

                            // Show apply button
                            this.generateBtn.style.display = 'none';
                            this.applyBtn.style.display = '';
                        } catch (e) {
                            // If not valid JSON, leave as-is (might be explanation)
                            if (this.mode === 'explain') {
                                this.previewStatus.textContent = 'Explanation ready';
                            }
                        }
                        return;
                    } else if (line.startsWith('event: error')) {
                        this.previewStatus.textContent = 'Generation failed';
                        return;
                    } else if (line.startsWith('event: complete')) {
                        // Extract JSON from complete event
                        const nextLine = lines[lines.indexOf(line) + 1];
                        if (nextLine?.startsWith('data: ')) {
                            const jsonStr = nextLine.slice(6).replace(/\\n/g, '\n');
                            try {
                                this.generatedStatechart = JSON.parse(jsonStr);
                                this.previewContent.textContent = JSON.stringify(this.generatedStatechart, null, 2);
                                this.previewStatus.textContent = 'Generated successfully';

                                this.generateBtn.style.display = 'none';
                                this.applyBtn.style.display = '';
                            } catch (e) {
                                console.error('Failed to parse complete JSON:', e);
                            }
                        }
                        return;
                    }
                }
            }
        } catch (error) {
            console.error('Stream reading error:', error);
            this.previewStatus.textContent = `Error: ${error.message}`;
        }
    }

    apply() {
        if (!this.generatedStatechart) return;

        this.dispatchEvent(new CustomEvent('statechartGenerated', {
            detail: { statechart: this.generatedStatechart }
        }));

        this.close();
    }

    // Public method to set current statechart for improve/explain
    setCurrentStatechart(statechart) {
        this.currentStatechart = statechart;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AIGenerator };
}

// Make available globally for browser
if (typeof window !== 'undefined') {
    window.AIGenerator = AIGenerator;
}
