// Enhanced JavaScript for AFL Prediction System

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();
    
    // Initialize team logo hover effects
    initTeamLogoEffects();
    
    // Initialize prediction sliders if they exist
    initPredictionSliders();
    
    // Add animation classes to elements
    addAnimations();
    
    // Initialize charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        initCharts();
    }
});

// Initialize Bootstrap tooltips
function initTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add hover effects to team logos
function initTeamLogoEffects() {
    const teamLogos = document.querySelectorAll('.team-logo');
    teamLogos.forEach(logo => {
        logo.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.2)';
        });
        logo.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
        });
    });
}

// Initialize prediction sliders
function initPredictionSliders() {
    const sliders = document.querySelectorAll('.prediction-slider input');
    sliders.forEach(slider => {
        const valueDisplay = slider.parentElement.querySelector('.prediction-value');
        if (valueDisplay) {
            // Update value display on input change
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        }
    });
}

// Add animation classes to elements
function addAnimations() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.classList.add('fade-in');
        card.style.animationDelay = (index * 0.1) + 's';
    });
    
    // Add slide-in animation to prediction items
    const predictionItems = document.querySelectorAll('.list-group-item');
    predictionItems.forEach((item, index) => {
        item.classList.add('slide-in');
        item.style.animationDelay = (index * 0.1) + 's';
    });
}

// Initialize charts for data visualization
function initCharts() {
    // Player form chart
    const formChartCanvas = document.getElementById('playerFormChart');
    if (formChartCanvas) {
        const ctx = formChartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: formChartCanvas.dataset.labels ? JSON.parse(formChartCanvas.dataset.labels) : [],
                datasets: [{
                    label: 'Disposals',
                    data: formChartCanvas.dataset.values ? JSON.parse(formChartCanvas.dataset.values) : [],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // Prediction accuracy chart
    const accuracyChartCanvas = document.getElementById('predictionAccuracyChart');
    if (accuracyChartCanvas) {
        const ctx = accuracyChartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: accuracyChartCanvas.dataset.labels ? JSON.parse(accuracyChartCanvas.dataset.labels) : [],
                datasets: [{
                    label: 'Prediction Accuracy',
                    data: accuracyChartCanvas.dataset.values ? JSON.parse(accuracyChartCanvas.dataset.values) : [],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(255, 193, 7, 0.7)',
                        'rgba(220, 53, 69, 0.7)'
                    ],
                    borderColor: [
                        'rgb(40, 167, 69)',
                        'rgb(255, 193, 7)',
                        'rgb(220, 53, 69)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
}

// Function to load team logos dynamically
function loadTeamLogos() {
    fetch('/static/images/teams/teams.json')
        .then(response => response.json())
        .then(teams => {
            const teamElements = document.querySelectorAll('[data-team]');
            teamElements.forEach(element => {
                const teamId = element.dataset.team;
                if (teams[teamId]) {
                    const logoPath = `/static/images/teams/${teamId}.svg`;
                    const img = document.createElement('img');
                    img.src = logoPath;
                    img.alt = teams[teamId].name;
                    img.className = 'team-logo';
                    element.appendChild(img);
                }
            });
        })
        .catch(error => console.error('Error loading team logos:', error));
}

// Function to toggle prediction details
function togglePredictionDetails(predictionId) {
    const detailsElement = document.getElementById(`prediction-details-${predictionId}`);
    if (detailsElement) {
        if (detailsElement.style.display === 'none') {
            detailsElement.style.display = 'block';
            detailsElement.classList.add('slide-in');
        } else {
            detailsElement.style.display = 'none';
        }
    }
}

// Function to save user prediction
function saveUserPrediction(predictionId) {
    // Get CSRF token from meta tag
    const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    
    fetch('/save_prediction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({
            prediction_id: predictionId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            const saveButton = document.querySelector(`button[data-prediction-id="${predictionId}"]`);
            if (saveButton) {
                saveButton.innerHTML = '<i class="fas fa-check"></i> Saved';
                saveButton.classList.remove('btn-outline-primary');
                saveButton.classList.add('btn-success');
                
                // Reset after 3 seconds
                setTimeout(() => {
                    saveButton.innerHTML = '<i class="fas fa-bookmark"></i> Save';
                    saveButton.classList.remove('btn-success');
                    saveButton.classList.add('btn-outline-primary');
                }, 3000);
            }
        } else {
            console.error('Error saving prediction:', data.error);
        }
    })
    .catch(error => console.error('Error saving prediction:', error));
}

// Load team logos when the page loads
document.addEventListener('DOMContentLoaded', loadTeamLogos);
