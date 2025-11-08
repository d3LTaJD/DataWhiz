// Modern JavaScript for DataWhiz Electron App
const { ipcRenderer } = require('electron');

class DataWhizApp {
    constructor() {
        this.currentSection = 'dashboard';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupNavigation();
        this.setupAnimations();
        this.loadUserPreferences();
    }

    setupEventListeners() {
        // Navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                this.navigateToSection(section);
            });
        });

        // Launch dashboard button
        document.getElementById('launch-dashboard').addEventListener('click', () => {
            this.launchDashboard();
        });

        // Quick start button
        document.getElementById('quick-start').addEventListener('click', () => {
            this.showQuickStartGuide();
        });

        // Feature cards
        document.querySelectorAll('.feature-card').forEach(card => {
            card.addEventListener('click', () => {
                const feature = card.dataset.feature;
                this.showFeatureDetails(feature);
            });
        });

        // Action cards
        document.querySelectorAll('.action-card').forEach(card => {
            card.addEventListener('click', () => {
                const action = card.id;
                this.handleQuickAction(action);
            });
        });

        // Search functionality
        const searchInput = document.querySelector('.search-input');
        searchInput.addEventListener('input', (e) => {
            this.handleSearch(e.target.value);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    setupNavigation() {
        // Set active navigation item
        this.updateActiveNavItem('dashboard');
    }

    setupAnimations() {
        // Add entrance animations
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        });

        document.querySelectorAll('.feature-card, .action-card, .project-item').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        });
    }

    navigateToSection(section) {
        this.currentSection = section;
        this.updateActiveNavItem(section);
        this.showSectionContent(section);
    }

    updateActiveNavItem(section) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeItem = document.querySelector(`[data-section="${section}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
    }

    showSectionContent(section) {
        // This would show different content based on the section
        console.log(`Navigating to: ${section}`);
        
        // For now, just show a message
        this.showNotification(`Switched to ${section} section`);
    }

    async launchDashboard() {
        try {
            // Show loading overlay
            this.showLoadingOverlay();
            
            // Launch the main dashboard
            const result = await ipcRenderer.invoke('launch-dashboard');
            
            if (result.success) {
                this.showNotification('Dashboard launched successfully!', 'success');
                
                // Close the landing page after a delay
                setTimeout(() => {
                    window.close();
                }, 2000);
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            this.showNotification(`Error launching dashboard: ${error.message}`, 'error');
        } finally {
            this.hideLoadingOverlay();
        }
    }

    showQuickStartGuide() {
        this.showModal('Quick Start Guide', `
            <div class="quick-start-content">
                <h3>Welcome to DataWhiz!</h3>
                <p>Follow these steps to get started:</p>
                <ol>
                    <li><strong>Upload Data:</strong> Click "Upload Data" to import your CSV, Excel, or JSON files</li>
                    <li><strong>Explore:</strong> Use the data table to explore your dataset</li>
                    <li><strong>Analyze:</strong> Run statistical analysis and create visualizations</li>
                    <li><strong>Machine Learning:</strong> Build and train ML models</li>
                </ol>
                <p>Ready to start? Click "Launch Full Dashboard" to begin!</p>
            </div>
        `);
    }

    showFeatureDetails(feature) {
        const featureInfo = {
            'data-management': {
                title: 'Data Management',
                description: 'Comprehensive data import, cleaning, and preprocessing tools.',
                features: [
                    'Support for CSV, Excel, JSON, Parquet files',
                    'Interactive data exploration',
                    'Missing value handling',
                    'Data type conversion',
                    'Outlier detection and treatment'
                ]
            },
            'statistical-analysis': {
                title: 'Statistical Analysis',
                description: 'Advanced statistical methods and hypothesis testing.',
                features: [
                    'Descriptive statistics',
                    'Correlation analysis',
                    'Hypothesis testing',
                    'Regression analysis',
                    'Time series analysis'
                ]
            },
            'visualization': {
                title: 'Advanced Visualizations',
                description: 'Interactive charts and customizable visualizations.',
                features: [
                    'Interactive Plotly charts',
                    'Statistical plots',
                    'Custom themes',
                    'Export capabilities',
                    'Real-time updates'
                ]
            },
            'machine-learning': {
                title: 'Machine Learning',
                description: 'Complete ML pipeline from data to deployment.',
                features: [
                    'Classification algorithms',
                    'Regression models',
                    'Clustering analysis',
                    'Model evaluation',
                    'Hyperparameter tuning'
                ]
            }
        };

        const info = featureInfo[feature];
        if (info) {
            this.showModal(info.title, `
                <div class="feature-details">
                    <p class="feature-description">${info.description}</p>
                    <h4>Key Features:</h4>
                    <ul>
                        ${info.features.map(f => `<li>${f}</li>`).join('')}
                    </ul>
                </div>
            `);
        }
    }

    handleQuickAction(action) {
        const actions = {
            'upload-data': () => this.showNotification('Opening file upload dialog...', 'info'),
            'create-analysis': () => this.navigateToSection('analysis'),
            'build-visualization': () => this.navigateToSection('visualization'),
            'train-model': () => this.navigateToSection('ml')
        };

        if (actions[action]) {
            actions[action]();
        }
    }

    handleSearch(query) {
        if (query.length < 2) return;
        
        // Simulate search functionality
        console.log(`Searching for: ${query}`);
        this.showNotification(`Searching for "${query}"...`, 'info');
    }

    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + K for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            document.querySelector('.search-input').focus();
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            this.closeModal();
        }
    }

    showLoadingOverlay() {
        document.getElementById('loading-overlay').classList.add('show');
    }

    hideLoadingOverlay() {
        document.getElementById('loading-overlay').classList.remove('show');
    }

    showModal(title, content) {
        // Create modal if it doesn't exist
        let modal = document.getElementById('modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'modal';
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-overlay">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3 class="modal-title"></h3>
                            <button class="modal-close">&times;</button>
                        </div>
                        <div class="modal-body"></div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
            
            // Add event listeners
            modal.querySelector('.modal-close').addEventListener('click', () => this.closeModal());
            modal.querySelector('.modal-overlay').addEventListener('click', (e) => {
                if (e.target === e.currentTarget) this.closeModal();
            });
        }
        
        // Update modal content
        modal.querySelector('.modal-title').textContent = title;
        modal.querySelector('.modal-body').innerHTML = content;
        
        // Show modal
        modal.style.display = 'flex';
        setTimeout(() => modal.classList.add('show'), 10);
    }

    closeModal() {
        const modal = document.getElementById('modal');
        if (modal) {
            modal.classList.remove('show');
            setTimeout(() => modal.style.display = 'none', 300);
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Show notification
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    getNotificationIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-circle',
            'warning': 'exclamation-triangle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    loadUserPreferences() {
        // Load user preferences from localStorage
        const preferences = localStorage.getItem('datawhiz-preferences');
        if (preferences) {
            try {
                const prefs = JSON.parse(preferences);
                // Apply preferences
                console.log('Loaded user preferences:', prefs);
            } catch (error) {
                console.error('Error loading preferences:', error);
            }
        }
    }

    saveUserPreferences(preferences) {
        // Save user preferences to localStorage
        localStorage.setItem('datawhiz-preferences', JSON.stringify(preferences));
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dataWhizApp = new DataWhizApp();
});

// Add modal styles
const modalStyles = `
.modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.8);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.modal.show {
    opacity: 1;
}

.modal-content {
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: var(--shadow-hover);
}

.modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-color);
}

.modal-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}

.modal-close {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: var(--transition);
}

.modal-close:hover {
    background-color: var(--tertiary-bg);
    color: var(--text-primary);
}

.modal-body {
    padding: 24px;
}

.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background-color: var(--secondary-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 12px 16px;
    box-shadow: var(--shadow);
    z-index: 10001;
    opacity: 0;
    transform: translateX(100%);
    transition: all 0.3s ease;
}

.notification.show {
    opacity: 1;
    transform: translateX(0);
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary);
}

.notification-success {
    border-color: var(--success-color);
}

.notification-error {
    border-color: var(--warning-color);
}

.notification-warning {
    border-color: #f59e0b;
}

.notification-info {
    border-color: var(--accent-color);
}
`;

// Add modal styles to the page
const styleSheet = document.createElement('style');
styleSheet.textContent = modalStyles;
document.head.appendChild(styleSheet);
