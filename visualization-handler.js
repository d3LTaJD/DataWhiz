/**
 * Enhanced Visualization Handler for DataWhiz Analytics
 * Handles dynamic chart creation based on analysis results
 */

class VisualizationHandler {
    constructor() {
        this.chartInstances = new Map();
        this.colorPalette = [
            '#58a6ff', '#3fb950', '#f85149', '#c9d1d9', '#d29922', 
            '#a371f7', '#ff7b72', '#8b949e', '#f0f6fc', '#6e7681'
        ];
    }

    /**
     * Create visualization based on analysis type and data
     */
    async createVisualization(containerId, analysisType, data) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            // Try to find container by partial ID match
            const partialMatch = document.querySelector(`[id*="${containerId.split('-').pop()}"]`);
            if (partialMatch) {
                console.log(`Found partial match: ${partialMatch.id}`);
                return this.createVisualization(partialMatch.id, analysisType, data);
            }
            return;
        }

        // Clear existing content
        container.innerHTML = '';

        try {
            switch (analysisType) {
                case 'total_revenue':
                case 'avg_order_value':
                    await this.createRevenueVisualization(container, data);
                    break;
                case 'revenue_trends':
                case 'daily_patterns':
                    await this.createTrendsVisualization(container, data);
                    break;
                case 'rfm_analysis':
                case 'customer_lifetime_value':
                    await this.createRFMVisualization(container, data);
                    break;
                case 'top_products':
                case 'category_analysis':
                    await this.createProductVisualization(container, data);
                    break;
                case 'revenue_by_country':
                case 'geographic_analysis':
                    await this.createGeographicVisualization(container, data);
                    break;
                case 'demand_forecasting':
                    await this.createMLVisualization(container, data, 'regression');
                    break;
                case 'churn_prediction':
                    await this.createMLVisualization(container, data, 'classification');
                    break;
                case 'customer_segmentation':
                    await this.createMLVisualization(container, data, 'clustering');
                    break;
                case 'anomaly_detection':
                    await this.createMLVisualization(container, data, 'anomaly');
                    break;
                default:
                    await this.createGenericVisualization(container, data);
            }
        } catch (error) {
            console.error('Error creating visualization:', error);
            container.innerHTML = `<div class="error-message">Error creating visualization: ${error.message}</div>`;
        }
    }

    /**
     * Create revenue analysis visualization
     */
    async createRevenueVisualization(container, data) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create chart type selector
        const selector = this.createChartTypeSelector(viz.chart_types);
        chartContainer.appendChild(selector);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `revenue-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        // Create chart with proper data validation
        const chartData = {
            labels: Array.isArray(viz.chart_data.labels) ? viz.chart_data.labels : [],
            datasets: [{
                label: 'Revenue Metrics',
                data: Array.isArray(viz.chart_data.values) ? viz.chart_data.values : [],
                backgroundColor: this.colorPalette.slice(0, viz.chart_data.labels.length),
                borderColor: this.colorPalette.slice(0, viz.chart_data.labels.length),
                borderWidth: 2
            }]
        };

        const chart = this.createSafeChart(canvas, {
            type: 'bar',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: viz.chart_title || 'Chart',
                        color: '#c9d1d9',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#8b949e' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' }
                    },
                    y: { 
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' }
                    }
                }
            }
        });

        // Store chart instance
        this.chartInstances.set(canvas.id, chart);

        // Add chart type change handler
        selector.addEventListener('change', (e) => {
            chart.destroy();
            const newConfig = {
                type: e.target.value,
                data: chart.data,
                options: chart.options
            };
            
            // Handle pie and doughnut charts differently
            if (e.target.value === 'pie' || e.target.value === 'doughnut') {
                newConfig.data = {
                    labels: viz.chart_data.labels,
                    datasets: [{
                        data: viz.chart_data.values,
                        backgroundColor: this.colorPalette.slice(0, viz.chart_data.labels.length),
                        borderColor: '#0d1117',
                        borderWidth: 2
                    }]
                };
                newConfig.options = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: viz.chart_title || 'Chart',
                            color: '#c9d1d9',
                            font: { size: 16 }
                        },
                        legend: {
                            labels: { color: '#8b949e' }
                        }
                    }
                };
            }
            
            const newChart = this.createSafeChart(canvas, newConfig);
            this.chartInstances.set(canvas.id, newChart);
        });
    }

    /**
     * Create trends visualization
     */
    async createTrendsVisualization(container, data) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create tabs for different time periods
        const tabs = this.createTabs(['Daily', 'Weekly', 'Monthly']);
        chartContainer.appendChild(tabs);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `trends-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        // Create initial daily chart with data validation
        const dailyData = {
            labels: Array.isArray(viz.daily_chart.labels) ? viz.daily_chart.labels : [],
            datasets: [{
                label: 'Revenue',
                data: Array.isArray(viz.daily_chart.data) ? viz.daily_chart.data : [],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        };

        const chart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: dailyData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: viz.daily_chart.title || 'Daily Trends',
                        color: '#c9d1d9',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#8b949e' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' }
                    },
                    y: { 
                        ticks: { color: '#8b949e' },
                        grid: { color: '#30363d' }
                    }
                }
            }
        });

        this.chartInstances.set(canvas.id, chart);

        // Add tab change handlers
        tabs.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab')) {
                const period = e.target.textContent.toLowerCase();
                const chartData = viz[`${period}_chart`];
                
                if (chartData && chartData.labels && chartData.data) {
                    chart.data.labels = chartData.labels;
                    chart.data.datasets[0].data = chartData.data;
                    chart.options.plugins.title.text = chartData.title;
                    chart.update();
                } else {
                    console.warn(`No data available for ${period} chart`);
                    // Use fallback data
                    chart.data.labels = ['No Data'];
                    chart.data.datasets[0].data = [0];
                    chart.options.plugins.title.text = `${period.charAt(0).toUpperCase() + period.slice(1)} Trends - No Data`;
                    chart.update();
                }
            }
        });
    }

    /**
     * Create RFM analysis visualization
     */
    async createRFMVisualization(container, data) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create chart type selector
        const selector = this.createChartTypeSelector(viz.chart_types);
        chartContainer.appendChild(selector);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `rfm-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        // Create scatter plot for RFM analysis
        const chart = new Chart(canvas.getContext('2d'), {
            type: 'scatter',
            data: {
                datasets: viz.scatter_data.map((point, index) => ({
                    label: point.Segment,
                    data: [{
                        x: point.Recency,
                        y: point.Frequency,
                        r: Math.sqrt(point.Monetary) / 10 // Bubble size based on monetary value
                    }],
                    backgroundColor: this.getSegmentColor(point.Segment),
                    borderColor: this.getSegmentColor(point.Segment),
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'RFM Customer Segments',
                        color: '#c9d1d9',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#8b949e' }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Recency (Days)', color: '#8b949e' },
                        ticks: { color: '#8b949e' }
                    },
                    y: {
                        title: { display: true, text: 'Frequency', color: '#8b949e' },
                        ticks: { color: '#8b949e' }
                    }
                }
            }
        });

        this.chartInstances.set(canvas.id, chart);

        // Add chart type change handler
        selector.addEventListener('change', (e) => {
            chart.destroy();
            let newConfig;
            
            if (e.target.value === 'pie' || e.target.value === 'doughnut') {
                newConfig = {
                    type: e.target.value,
                    data: {
                        labels: viz.segment_pie.labels,
                        datasets: [{
                            data: viz.segment_pie.data,
                            backgroundColor: viz.segment_pie.colors,
                            borderColor: '#0d1117',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Customer Segment Distribution',
                                color: '#c9d1d9',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: '#8b949e' }
                            }
                        }
                    }
                };
            } else {
                // For other chart types, use the original scatter plot data
                newConfig = {
                    type: e.target.value,
                    data: chart.data,
                    options: chart.options
                };
            }
            
            const newChart = this.createSafeChart(canvas, newConfig);
            this.chartInstances.set(canvas.id, newChart);
        });
    }

    /**
     * Create product analysis visualization
     */
    async createProductVisualization(container, data) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create chart type selector
        const selector = this.createChartTypeSelector(viz.chart_types);
        chartContainer.appendChild(selector);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `product-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        // Create horizontal bar chart for top products
        const chart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: viz.top_products_chart.labels,
                datasets: [{
                    label: 'Revenue',
                    data: viz.top_products_chart.revenue_data,
                    backgroundColor: this.colorPalette[0],
                    borderColor: this.colorPalette[0],
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: viz.top_products_chart.title,
                        color: '#c9d1d9',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#8b949e' }
                    }
                },
                scales: {
                    x: { ticks: { color: '#8b949e' } },
                    y: { ticks: { color: '#8b949e' } }
                }
            }
        });

        this.chartInstances.set(canvas.id, chart);
    }

    /**
     * Create geographic analysis visualization
     */
    async createGeographicVisualization(container, data) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create chart type selector
        const selector = this.createChartTypeSelector(viz.chart_types);
        chartContainer.appendChild(selector);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `geo-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        // Create bar chart for country revenue
        const chart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: viz.country_revenue_chart.labels,
                datasets: [{
                    label: 'Revenue',
                    data: viz.country_revenue_chart.data,
                    backgroundColor: this.colorPalette.slice(0, viz.country_revenue_chart.labels.length),
                    borderColor: this.colorPalette.slice(0, viz.country_revenue_chart.labels.length),
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: viz.country_revenue_chart.title,
                        color: '#c9d1d9',
                        font: { size: 16 }
                    },
                    legend: {
                        labels: { color: '#8b949e' }
                    }
                },
                scales: {
                    x: { ticks: { color: '#8b949e' } },
                    y: { ticks: { color: '#8b949e' } }
                }
            }
        });

        this.chartInstances.set(canvas.id, chart);
    }

    /**
     * Create ML model visualization
     */
    async createMLVisualization(container, data, modelType) {
        if (!data.visualization) {
            this.createFallbackVisualization(container, data);
            return;
        }

        const viz = data.visualization;
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        
        // Create chart type selector
        const selector = this.createChartTypeSelector(viz.chart_types);
        chartContainer.appendChild(selector);

        // Create canvas
        const canvas = document.createElement('canvas');
        canvas.id = `ml-chart-${Date.now()}`;
        chartContainer.appendChild(canvas);

        container.appendChild(chartContainer);

        let chart;
        
        switch (modelType) {
            case 'regression':
                // Create actual vs predicted chart
                chart = new Chart(canvas.getContext('2d'), {
                    type: 'scatter',
                    data: {
                        labels: viz.actual_vs_predicted.labels,
                        datasets: [{
                            label: 'Actual',
                            data: viz.actual_vs_predicted.actual_data.map((val, i) => ({x: i, y: val})),
                            backgroundColor: '#58a6ff',
                            borderColor: '#58a6ff'
                        }, {
                            label: 'Predicted',
                            data: viz.actual_vs_predicted.predicted_data.map((val, i) => ({x: i, y: val})),
                            backgroundColor: '#3fb950',
                            borderColor: '#3fb950'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: viz.actual_vs_predicted.title,
                                color: '#c9d1d9',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: '#8b949e' }
                            }
                        },
                        scales: {
                            x: { ticks: { color: '#8b949e' } },
                            y: { ticks: { color: '#8b949e' } }
                        }
                    }
                });
                break;
                
            case 'classification':
                // Create confusion matrix visualization
                chart = new Chart(canvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: viz.metrics_chart.labels,
                        datasets: [{
                            label: 'Score',
                            data: viz.metrics_chart.data,
                            backgroundColor: this.colorPalette.slice(0, viz.metrics_chart.labels.length),
                            borderColor: this.colorPalette.slice(0, viz.metrics_chart.labels.length),
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: viz.metrics_chart.title,
                                color: '#c9d1d9',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: '#8b949e' }
                            }
                        },
                        scales: {
                            x: { ticks: { color: '#8b949e' } },
                            y: { 
                                ticks: { color: '#8b949e' },
                                min: 0,
                                max: 1
                            }
                        }
                    }
                });
                break;
                
            case 'clustering':
                // Create cluster scatter plot
                chart = new Chart(canvas.getContext('2d'), {
                    type: 'scatter',
                    data: {
                        datasets: viz.scatter_plot.data.map((point, index) => ({
                            label: `Cluster ${point.Cluster}`,
                            data: [{
                                x: point.Monetary,
                                y: point.Frequency
                            }],
                            backgroundColor: this.colorPalette[point.Cluster % this.colorPalette.length],
                            borderColor: this.colorPalette[point.Cluster % this.colorPalette.length]
                        }))
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: viz.scatter_plot.title,
                                color: '#c9d1d9',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: '#8b949e' }
                            }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: viz.scatter_plot.x_label, color: '#8b949e' },
                                ticks: { color: '#8b949e' }
                            },
                            y: {
                                title: { display: true, text: viz.scatter_plot.y_label, color: '#8b949e' },
                                ticks: { color: '#8b949e' }
                            }
                        }
                    }
                });
                break;
                
            case 'anomaly':
                // Create anomaly distribution chart
                chart = new Chart(canvas.getContext('2d'), {
                    type: 'pie',
                    data: {
                        labels: viz.anomaly_distribution.labels,
                        datasets: [{
                            data: viz.anomaly_distribution.data,
                            backgroundColor: ['#3fb950', '#f85149'],
                            borderColor: '#0d1117',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: viz.anomaly_distribution.title,
                                color: '#c9d1d9',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: '#8b949e' }
                            }
                        }
                    }
                });
                break;
        }

        this.chartInstances.set(canvas.id, chart);
    }

    /**
     * Create generic visualization fallback
     */
    createFallbackVisualization(container, data) {
        container.innerHTML = `
            <div class="fallback-visualization">
                <h3>Analysis Results</h3>
                <div class="metrics-grid">
                    ${Object.entries(data).map(([key, value]) => 
                        `<div class="metric-card">
                            <div class="metric-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                            <div class="metric-value">${typeof value === 'object' ? JSON.stringify(value) : value}</div>
                        </div>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Create chart type selector
     */
    createChartTypeSelector(types) {
        const selector = document.createElement('select');
        selector.className = 'chart-type-selector';
        selector.style.cssText = `
            background: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 15px;
            font-size: 14px;
        `;
        
        if (Array.isArray(types)) {
            types.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type.charAt(0).toUpperCase() + type.slice(1);
                selector.appendChild(option);
            });
        }
        
        return selector;
    }

    /**
     * Safe chart creation with error handling
     */
    createSafeChart(canvas, config) {
        try {
            // Validate data before creating chart
            if (!config.data || !Array.isArray(config.data.labels)) {
                console.warn('Invalid chart data, using fallback');
                config.data = {
                    labels: ['No Data'],
                    datasets: [{
                        label: 'No Data',
                        data: [0],
                        backgroundColor: '#8b949e'
                    }]
                };
            }

            // Ensure all data arrays are valid
            if (config.data.datasets) {
                config.data.datasets.forEach(dataset => {
                    if (!Array.isArray(dataset.data)) {
                        dataset.data = [0];
                    }
                });
            }

            return new Chart(canvas.getContext('2d'), config);
        } catch (error) {
            console.error('Chart creation failed:', error);
            // Create a simple error chart
            return new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['Error'],
                    datasets: [{
                        label: 'Chart Error',
                        data: [0],
                        backgroundColor: '#f85149'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Chart Error',
                            color: '#f85149'
                        }
                    }
                }
            });
        }
    }

    /**
     * Create tabs for different views
     */
    createTabs(tabNames) {
        const tabsContainer = document.createElement('div');
        tabsContainer.className = 'tabs-container';
        tabsContainer.style.cssText = `
            display: flex;
            margin-bottom: 15px;
            border-bottom: 1px solid #30363d;
        `;
        
        tabNames.forEach((name, index) => {
            const tab = document.createElement('button');
            tab.className = `tab ${index === 0 ? 'active' : ''}`;
            tab.textContent = name;
            tab.style.cssText = `
                background: ${index === 0 ? '#58a6ff' : 'transparent'};
                color: ${index === 0 ? '#0d1117' : '#c9d1d9'};
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                border-radius: 6px 6px 0 0;
                font-size: 14px;
                transition: all 0.3s ease;
            `;
            tabsContainer.appendChild(tab);
        });
        
        return tabsContainer;
    }

    /**
     * Get segment color
     */
    getSegmentColor(segment) {
        const segmentColors = {
            'Champions': '#3fb950',
            'Loyal Customers': '#58a6ff',
            'Potential Loyalists': '#d29922',
            'New Customers': '#a371f7',
            'Promising': '#ff7b72',
            'Need Attention': '#f85149',
            'About to Sleep': '#8b949e',
            'At Risk': '#ffa657',
            'Cannot Lose Them': '#f0f6fc',
            'Hibernating': '#6e7681',
            'Lost': '#21262d'
        };
        return segmentColors[segment] || '#8b949e';
    }

    /**
     * Destroy all chart instances
     */
    destroyAllCharts() {
        this.chartInstances.forEach(chart => chart.destroy());
        this.chartInstances.clear();
    }
}

// Export for use in other files
window.VisualizationHandler = VisualizationHandler;
