document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const systemPrompt = document.getElementById('system-prompt');
    const templateSelect = document.getElementById('template-select');
    const saveTemplateBtn = document.getElementById('save-template-btn');
    const templateNameInput = document.getElementById('template-name');
    const toolSelectionContainer = document.getElementById('tool-selection-container');
    const responseArea = document.getElementById('response-area');
    const responseContent = document.getElementById('response-content');
    const responseMetadata = document.getElementById('response-metadata');
    const copyBtn = document.getElementById('copy-btn');
    const loader = document.getElementById('loader');
    const chartContainer = document.getElementById('chart-container');
    const updateTemplateBtn = document.getElementById('update-template-btn');
    
    // New elements
    const providerSelect = document.getElementById('provider-select');
    const modelSelect = document.getElementById('model-select');
    const providerStatus = document.getElementById('provider-status');
    const terminalSidebar = document.getElementById('terminal-sidebar');
    const terminalContent = document.getElementById('terminal-content');
    const toggleTerminal = document.getElementById('toggle-terminal');
    const terminalSpinner = document.getElementById('terminal-spinner');
    
    // API Key elements
    const apiKeySection = document.getElementById('api-key-section');
    const apiKeyInput = document.getElementById('api-key-input');
    const saveApiKeyBtn = document.getElementById('save-api-key-btn');
    const testApiKeyBtn = document.getElementById('test-api-key-btn');
    const apiKeyStatus = document.getElementById('api-key-status');
    const apiKeyHelp = document.getElementById('api-key-help');

    let currentProviders = {};
    let isProcessing = false;

    // --- Terminal Management ---
    function addTerminalLine(message, type = 'info') {
        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;
        line.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
        terminalContent.appendChild(line);
        terminalContent.scrollTop = terminalContent.scrollHeight;
        
        // Keep only last 50 lines
        while (terminalContent.children.length > 50) {
            terminalContent.removeChild(terminalContent.firstChild);
        }
    }

    function setTerminalActivity(active) {
        isProcessing = active;
        if (active) {
            terminalSpinner.classList.add('active');
        } else {
            terminalSpinner.classList.remove('active');
        }
    }

    toggleTerminal.addEventListener('click', () => {
        terminalSidebar.classList.toggle('collapsed');
        toggleTerminal.textContent = terminalSidebar.classList.contains('collapsed') ? '+' : '−';
    });

    // --- Provider Management ---
    async function loadProviders() {
        try {
            const response = await fetch('/providers');
            currentProviders = await response.json();
            
            // Populate provider select
            providerSelect.innerHTML = '';
            Object.entries(currentProviders.providers).forEach(([key, provider]) => {
                const option = new Option(provider.name, key);
                if (key === currentProviders.current_provider) {
                    option.selected = true;
                }
                providerSelect.add(option);
            });
            
            updateModelSelect();
            updateProviderStatus();
            updateApiKeySection();
            addTerminalLine(`Loaded providers: ${Object.keys(currentProviders.providers).join(', ')}`);
        } catch (error) {
            console.error('Failed to load providers:', error);
            addTerminalLine('Failed to load providers', 'error');
        }
    }

    function updateModelSelect() {
        const selectedProvider = providerSelect.value;
        const provider = currentProviders.providers[selectedProvider];
        
        modelSelect.innerHTML = '';
        if (provider && provider.models) {
            provider.models.forEach(model => {
                const option = new Option(model, model);
                if (model === currentProviders.current_model) {
                    option.selected = true;
                }
                modelSelect.add(option);
            });
        }
    }

    function updateProviderStatus() {
        const selectedProvider = providerSelect.value;
        const provider = currentProviders.providers[selectedProvider];
        
        if (selectedProvider === currentProviders.current_provider) {
            providerStatus.textContent = '✓ Connected';
            providerStatus.className = 'provider-status connected';
        } else {
            providerStatus.textContent = '⚠ Configuration needed';
            providerStatus.className = 'provider-status error';
        }
    }

    providerSelect.addEventListener('change', () => {
        updateModelSelect();
        updateProviderStatus();
        updateApiKeySection();
        addTerminalLine(`Provider changed to: ${providerSelect.value}`);
    });

    modelSelect.addEventListener('change', () => {
        addTerminalLine(`Model changed to: ${modelSelect.value}`);
    });

    // --- API Key Management ---
    function updateApiKeySection() {
        const selectedProvider = providerSelect.value;
        const provider = currentProviders.providers[selectedProvider];
        
        if (provider && provider.requires_key) {
            apiKeySection.style.display = 'block';
            
            // Update help text
            apiKeyHelp.innerHTML = `
                <strong>Get your ${provider.name} API key:</strong><br>
                1. Visit <a href="${provider.help_url}" target="_blank">${provider.help_url}</a><br>
                2. Create an account and generate an API key<br>
                3. Enter the key above and click "Save to .env"<br>
                4. Restart the Docker container for changes to take effect
            `;
            
            // Clear previous status
            apiKeyStatus.textContent = '';
            apiKeyStatus.className = 'api-key-status';
            apiKeyInput.value = '';
        } else {
            apiKeySection.style.display = 'none';
        }
    }

    saveApiKeyBtn.addEventListener('click', async () => {
        const provider = providerSelect.value;
        const apiKey = apiKeyInput.value.trim();
        
        if (!apiKey) {
            showApiKeyStatus('Please enter an API key', 'error');
            return;
        }
        
        try {
            setTerminalActivity(true);  // Show spinner during save
            showApiKeyStatus('Saving...', 'warning');
            addTerminalLine(`🔑 Saving API key for ${provider}...`, 'info');
            
            const response = await fetch('/api/save-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, api_key: apiKey })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                showApiKeyStatus('✅ Saved! Restart Docker to apply', 'success');
                addTerminalLine(`✅ API key saved for ${provider}`, 'success');
                apiKeyInput.value = '';
            } else {
                showApiKeyStatus(`❌ ${result.detail}`, 'error');
                addTerminalLine(`❌ Failed to save API key: ${result.detail}`, 'error');
            }
        } catch (error) {
            showApiKeyStatus(`❌ Save failed: ${error.message}`, 'error');
            addTerminalLine(`❌ API key save error: ${error.message}`, 'error');
        } finally {
            setTerminalActivity(false);  // Stop spinner
        }
    });

    testApiKeyBtn.addEventListener('click', async () => {
        const provider = providerSelect.value;
        const apiKey = apiKeyInput.value.trim();
        
        if (!apiKey) {
            showApiKeyStatus('Please enter an API key', 'error');
            return;
        }
        
        try {
            setTerminalActivity(true);  // Show spinner during test
            showApiKeyStatus('Testing...', 'warning');
            addTerminalLine(`🧪 Testing API key for ${provider}...`, 'info');
            
            const response = await fetch('/api/test-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, api_key: apiKey })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                showApiKeyStatus(`✅ ${result.message}`, 'success');
                addTerminalLine(`✅ API key test passed for ${provider}`, 'success');
            } else {
                showApiKeyStatus(`❌ ${result.message}`, 'error');
                addTerminalLine(`❌ API key test failed: ${result.message}`, 'error');
            }
        } catch (error) {
            showApiKeyStatus(`❌ Test failed: ${error.message}`, 'error');
            addTerminalLine(`❌ API key test error: ${error.message}`, 'error');
        } finally {
            setTerminalActivity(false);  // Stop spinner
        }
    });

    function showApiKeyStatus(message, type) {
        apiKeyStatus.textContent = message;
        apiKeyStatus.className = `api-key-status ${type}`;
    }

    // --- Chart Rendering Fix ---
    function renderChart(chartData) {
        try {
            console.log('Chart data received:', chartData);
            addTerminalLine('🔍 Processing chart data...', 'info');
            
            if (!chartData || !chartData.data) {
                console.error('Invalid chart data structure:', chartData);
                chartContainer.innerHTML = '<p class="error">Invalid chart data received</p>';
                addTerminalLine('❌ Invalid chart data structure', 'error');
                return;
            }

            // Clear previous chart
            chartContainer.innerHTML = '';
            
            // Handle OpenBB chart artifact format
            const data = chartData.data;
            const xKey = chartData.x_key || 'date';
            const ohlcKeys = chartData.ohlc_keys || {
                open: 'open',
                high: 'high', 
                low: 'low',
                close: 'close'
            };
            
            console.log('Data length:', data.length);
            console.log('X key:', xKey);
            console.log('OHLC Keys:', ohlcKeys);
            console.log('Sample data point:', data[0]);
            
            // Check if data has index-based date
            const firstRecord = data[0];
            const dateValue = firstRecord[xKey];
            console.log('Date value type:', typeof dateValue, 'Value:', dateValue);
            
            // Extract dates - handle both string dates and numeric indices
            let dates;
            if (typeof dateValue === 'string' || dateValue instanceof Date) {
                dates = data.map(d => d[xKey]);
                console.log('Using date strings:', dates.slice(0, 3));
            } else {
                // If no proper date field, create date sequence
                console.log('No proper dates found, creating date sequence...');
                const today = new Date();
                dates = data.map((_, index) => {
                    const date = new Date(today);
                    date.setDate(date.getDate() - (data.length - 1 - index));
                    return date.toISOString().split('T')[0];
                });
                console.log('Generated dates:', dates.slice(0, 3));
            }
            
            console.log('Price values sample:', data.slice(0, 3).map(d => ({
                open: d[ohlcKeys.open],
                high: d[ohlcKeys.high],
                low: d[ohlcKeys.low],
                close: d[ohlcKeys.close]
            })));
            
            // Ensure we have valid data
            if (!data || data.length === 0) {
                chartContainer.innerHTML = '<p class="error">No chart data available</p>';
                addTerminalLine('❌ No chart data points available', 'error');
                return;
            }
            
            addTerminalLine(`📊 Processing ${data.length} data points...`, 'info');
            
            // Extract and process data with better error handling
            const opens = data.map(d => Number(d[ohlcKeys.open]));
            const highs = data.map(d => Number(d[ohlcKeys.high]));
            const lows = data.map(d => Number(d[ohlcKeys.low]));
            const closes = data.map(d => Number(d[ohlcKeys.close]));
            
            console.log('Processed arrays:', {
                dates: dates.slice(0, 3),
                opens: opens.slice(0, 3),
                highs: highs.slice(0, 3),
                lows: lows.slice(0, 3),
                closes: closes.slice(0, 3)
            });
            
            // Check for NaN values
            const hasValidData = opens.every(v => !isNaN(v) && v > 0) && 
                                highs.every(v => !isNaN(v) && v > 0) && 
                                lows.every(v => !isNaN(v) && v > 0) && 
                                closes.every(v => !isNaN(v) && v > 0);
            
            if (!hasValidData) {
                addTerminalLine('❌ Invalid price data detected', 'error');
                console.error('Invalid price data detected');
                console.error('Opens:', opens.slice(0, 5));
                console.error('Highs:', highs.slice(0, 5));
                console.error('Lows:', lows.slice(0, 5));
                console.error('Closes:', closes.slice(0, 5));
                chartContainer.innerHTML = '<p class="error">Invalid price data detected</p>';
                return;
            }
            
            // Prepare data for Plotly candlestick chart
            const plotlyData = [{
                type: 'candlestick',
                x: dates,
                open: opens,
                high: highs,
                low: lows,
                close: closes,
                name: chartData.name || 'Price',
                increasing: { line: { color: '#26a69a' } },
                decreasing: { line: { color: '#ef5350' } },
                showlegend: false
            }];
            
            console.log('Final Plotly data structure:');
            console.log('X (dates):', plotlyData[0].x.slice(0, 3));
            console.log('Open:', plotlyData[0].open.slice(0, 3));
            console.log('High:', plotlyData[0].high.slice(0, 3));
            console.log('Low:', plotlyData[0].low.slice(0, 3));
            console.log('Close:', plotlyData[0].close.slice(0, 3));
            
            addTerminalLine('📈 Creating candlestick chart...', 'info');
            
            const layout = {
                title: {
                    text: chartData.name || 'Financial Chart',
                    font: { size: 16 }
                },
                xaxis: { 
                    title: 'Date',
                    type: 'date'
                },
                yaxis: { 
                    title: 'Price ($)',
                    autorange: true
                },
                margin: { t: 60, r: 50, b: 50, l: 60 },
                height: 450,
                plot_bgcolor: 'white',
                paper_bgcolor: 'white',
                showlegend: false,
                font: { family: 'Inter, sans-serif' }
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'toImage'],
                displaylogo: false
            };

            console.log('Creating Plotly chart with config:', { layout, config });
            addTerminalLine('🎨 Rendering chart with Plotly...', 'info');
            
            Plotly.newPlot(chartContainer, plotlyData, layout, config)
                .then(() => {
                    console.log('✅ Chart rendered successfully!');
                    addTerminalLine('✅ Chart rendered successfully!', 'success');
                    setTerminalActivity(false);  // Stop spinner when chart is done
                })
                .catch(error => {
                    console.error('Plotly error:', error);
                    chartContainer.innerHTML = `<p class="error">Chart rendering failed: ${error.message}</p>`;
                    addTerminalLine(`❌ Chart rendering failed: ${error.message}`, 'error');
                    setTerminalActivity(false);  // Stop spinner on error
                });
            
        } catch (error) {
            console.error('Error rendering chart:', error);
            console.error('Chart data was:', chartData);
            chartContainer.innerHTML = `<p class="error">Error rendering chart: ${error.message}</p>`;
            addTerminalLine(`❌ Chart error: ${error.message}`, 'error');
        }
    }

    // --- Fetch and display tools ---
    async function loadTools() {
        try {
            const response = await fetch('/tools');
            const tools = await response.json();
            
            toolSelectionContainer.innerHTML = '<h4>Permitted Agentic Tools:</h4>';
            tools.forEach(tool => {
                const checkbox = `
                    <div class="tool-checkbox">
                        <input type="checkbox" id="${tool.name}" name="${tool.name}" value="${tool.name}" checked>
                        <label for="${tool.name}" title="${tool.description}">${tool.name}</label>
                    </div>
                `;
                toolSelectionContainer.innerHTML += checkbox;
            });

            // Add prompt suggestions container
            toolSelectionContainer.innerHTML += `
                <div id="prompt-suggestions" class="prompt-suggestions">
                    <h5>Suggested Prompts for Selected Tools:</h5>
                    <div id="suggestions-list"></div>
                </div>
            `;

            // Add event listeners to checkboxes
            const checkboxes = toolSelectionContainer.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('change', updatePromptSuggestions);
            });

            updatePromptSuggestions();
            addTerminalLine(`Loaded ${tools.length} tools`);
        } catch (error) {
            console.error('Failed to load tools:', error);
            toolSelectionContainer.innerHTML = '<p class="error">Could not load tools.</p>';
            addTerminalLine('Failed to load tools', 'error');
        }
    }

    // --- Update prompt suggestions ---
    function updatePromptSuggestions() {
        const selectedTools = Array.from(toolSelectionContainer.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
        
        const suggestionsList = document.getElementById('suggestions-list');
        if (!suggestionsList) return;

        const suggestions = {
            'get_news': "Get the latest news for AAPL",
            'get_historical_price': "Provide historical price levels for Tesla over the last year",
            'generate_financial_chart': "Show me a financial chart for Microsoft stock"
        };

        if (selectedTools.length === 0) {
            suggestionsList.innerHTML = '<p class="no-suggestions">Select tools above to see prompt suggestions</p>';
            return;
        }

        const relevantSuggestions = selectedTools
            .filter(tool => suggestions[tool])
            .map(tool => `
                <div class="suggestion-item" onclick="insertSuggestion('${suggestions[tool]}')">
                    <strong>${tool}:</strong> "${suggestions[tool]}"
                </div>
            `);

        suggestionsList.innerHTML = relevantSuggestions.length > 0 
            ? relevantSuggestions.join('')
            : '<p class="no-suggestions">No specific suggestions for selected tools</p>';
    }

    // --- Insert suggestion into query input ---
    window.insertSuggestion = function(suggestion) {
        queryInput.value = suggestion;
        queryInput.focus();
        addTerminalLine(`Inserted suggestion: ${suggestion}`);
    };

    // --- Template Management ---
    const templates = JSON.parse(localStorage.getItem('promptTemplates') || '{}');
    if (!templates['Neutral']) {
        templates['Neutral'] = 'You are a helpful financial assistant.';
        localStorage.setItem('promptTemplates', JSON.stringify(templates));
    }

    function loadTemplates() {
        const currentTemplates = JSON.parse(localStorage.getItem('promptTemplates') || '{}');
        templateSelect.innerHTML = '';
        for (const name in currentTemplates) {
            const option = new Option(name, name);
            templateSelect.add(option);
        }
        templateSelect.value = 'Neutral';
        systemPrompt.value = currentTemplates['Neutral'];
    }

    updateTemplateBtn.addEventListener('click', () => {
        const name = templateSelect.value;
        const prompt = systemPrompt.value.trim();
        if (name && prompt) {
            const templates = JSON.parse(localStorage.getItem('promptTemplates') || '{}');
            templates[name] = prompt;
            localStorage.setItem('promptTemplates', JSON.stringify(templates));
            addTerminalLine(`Template '${name}' updated`);
        }
    });

    saveTemplateBtn.addEventListener('click', () => {
        const name = templateNameInput.value.trim();
        const prompt = systemPrompt.value.trim();
        if (name && prompt) {
            const templates = JSON.parse(localStorage.getItem('promptTemplates') || '{}');
            templates[name] = prompt;
            localStorage.setItem('promptTemplates', JSON.stringify(templates));
            loadTemplates();
            templateNameInput.value = '';
            addTerminalLine(`Template '${name}' saved`);
        }
    });

    templateSelect.addEventListener('change', () => {
        const name = templateSelect.value;
        const templates = JSON.parse(localStorage.getItem('promptTemplates') || '{}');
        if (templates[name]) {
            systemPrompt.value = templates[name];
            addTerminalLine(`Template '${name}' loaded`);
        }
    });

    // --- Real-time Form Submission with SSE ---
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        const selectedTools = Array.from(toolSelectionContainer.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);

        // Clear previous response and show progress
        responseArea.style.display = 'none';
        chartContainer.innerHTML = '';
        terminalContent.innerHTML = '<div class="terminal-line">Starting analysis...</div>';
        setTerminalActivity(true);

        addTerminalLine(`Query: ${query}`);
        addTerminalLine(`Tools: ${selectedTools.join(', ') || 'None'}`);

        // Use fallback approach since EventSource doesn't support POST with body
        fallbackToRegularRequest(query, selectedTools);
    });

    // --- Fallback to non-streaming request ---
    async function fallbackToRegularRequest(query, selectedTools) {
        try {
            addTerminalLine('Falling back to non-streaming mode...', 'warning');
            
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query, 
                    selected_tools: selectedTools,
                    system_prompt: systemPrompt.value.trim() 
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Unknown error occurred');
            }

            const data = await response.json();
            handleFinalResponse(data);

        } catch (error) {
            addTerminalLine(`Fallback error: ${error.message}`, 'error');
            responseContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            responseMetadata.innerHTML = '';
            responseArea.style.display = 'block';
        } finally {
            setTerminalActivity(false);
        }
    }

    // --- Handle final response ---
    function handleFinalResponse(data) {
        // Render chart if available
        if (data.chart) {
            addTerminalLine('📊 Chart data received, rendering...', 'info');
            setTerminalActivity(true);  // Show spinner during chart rendering
            renderChart(data.chart);
        } else {
            addTerminalLine('ℹ️ No chart data in response', 'warning');
        }

        // Render markdown content
        const formattedContent = marked.parse(data.content);
        responseContent.innerHTML = formattedContent;
        
        // Display metadata
        const toolsUsedHtml = data.tools_used.length > 0
            ? data.tools_used.map(tool => `<span class="tool-tag">${tool}</span>`).join(' ')
            : '<span>None</span>';

        const provider_model_info = `${data.llm_provider} (${data.model})`;

        responseMetadata.innerHTML = `
            <div class="metadata-item">
                <strong>Tools Activated:</strong> ${toolsUsedHtml}
            </div>
            <div class="metadata-item">
                <strong>Provider:</strong> <span>${provider_model_info} (${data.token_usage} tokens)</span>
            </div>
        `;
        
        responseArea.style.display = 'block';
        addTerminalLine('✅ Analysis complete!', 'success');
        setTerminalActivity(false);
    }

    // --- Copy functionality ---
    copyBtn.addEventListener('click', () => {
        const fullHtml = `
            <div style="font-family: sans-serif; line-height: 1.6;">
                ${responseContent.innerHTML}
                <hr>
                <div style="font-size: 0.8em; color: #555;">
                    ${responseMetadata.innerHTML}
                </div>
            </div>
        `;
        const blob = new Blob([fullHtml], { type: 'text/html' });
        navigator.clipboard.write([new ClipboardItem({ 'text/html': blob })])
            .then(() => {
                copyBtn.querySelector('svg').style.fill = 'green';
                setTimeout(() => {
                    copyBtn.querySelector('svg').style.fill = 'currentColor';
                }, 1500);
                addTerminalLine('Response copied to clipboard');
            });
    });

    // --- Initialize everything ---
    loadProviders();
    loadTools();
    loadTemplates();
    addTerminalLine('🚀 OpenBB Agent initialized');
}); 