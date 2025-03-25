// Main JavaScript for AFL Prediction System

document.addEventListener('DOMContentLoaded', function() {
    // Enable tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Enable popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    });

    // Auto-dismiss alerts
    setTimeout(function() {
        $('.alert').alert('close');
    }, 5000);

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Player search functionality
    const playerSearch = document.getElementById('playerSearch');
    if (playerSearch) {
        playerSearch.addEventListener('keyup', function() {
            const searchTerm = this.value.toLowerCase();
            const playerRows = document.querySelectorAll('.player-row');
            
            playerRows.forEach(row => {
                const playerName = row.querySelector('.player-name').textContent.toLowerCase();
                const playerTeam = row.querySelector('.player-team').textContent.toLowerCase();
                
                if (playerName.includes(searchTerm) || playerTeam.includes(searchTerm)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }

    // Team filter functionality
    const teamFilter = document.getElementById('teamFilter');
    if (teamFilter) {
        teamFilter.addEventListener('change', function() {
            const selectedTeam = this.value;
            const playerRows = document.querySelectorAll('.player-row');
            
            playerRows.forEach(row => {
                const playerTeam = row.querySelector('.player-team').textContent;
                
                if (selectedTeam === 'all' || playerTeam === selectedTeam) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }

    // Fetch latest predictions via API
    const predictionContainer = document.getElementById('latestPredictions');
    if (predictionContainer) {
        fetch('/api/predictions')
            .then(response => response.json())
            .then(data => {
                if (data.length > 0) {
                    const topPredictions = data.slice(0, 5);
                    let html = '<ul class="list-group">';
                    
                    topPredictions.forEach(prediction => {
                        html += `
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>${prediction.player_name}</strong>
                                    <small class="d-block text-muted">${prediction.team} vs ${prediction.home_team === prediction.team ? prediction.away_team : prediction.home_team}</small>
                                </div>
                                <span class="badge bg-primary rounded-pill">${prediction.predicted_disposals.toFixed(1)}</span>
                            </li>
                        `;
                    });
                    
                    html += '</ul>';
                    predictionContainer.innerHTML = html;
                } else {
                    predictionContainer.innerHTML = '<p class="text-muted">No predictions available.</p>';
                }
            })
            .catch(error => {
                console.error('Error fetching predictions:', error);
                predictionContainer.innerHTML = '<p class="text-danger">Error loading predictions.</p>';
            });
    }

    // Create charts for player form
    const playerFormCharts = document.querySelectorAll('.player-form-chart');
    if (playerFormCharts.length > 0) {
        playerFormCharts.forEach(chartCanvas => {
            const playerId = chartCanvas.dataset.playerId;
            
            fetch(`/api/players/${playerId}`)
                .then(response => response.json())
                .then(player => {
                    const ctx = chartCanvas.getContext('2d');
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: ['Last 3 Disposals', 'Last 5 Disposals', 'Avg Disposals', 'Last 3 Goals', 'Last 5 Goals', 'Avg Goals'],
                            datasets: [{
                                label: 'Player Form',
                                data: [
                                    player.last_3_disposals,
                                    player.last_5_disposals,
                                    player.avg_disposals,
                                    player.last_3_goals,
                                    player.last_5_goals,
                                    player.avg_goals
                                ],
                                backgroundColor: [
                                    'rgba(54, 162, 235, 0.7)',
                                    'rgba(54, 162, 235, 0.5)',
                                    'rgba(54, 162, 235, 0.3)',
                                    'rgba(255, 99, 132, 0.7)',
                                    'rgba(255, 99, 132, 0.5)',
                                    'rgba(255, 99, 132, 0.3)'
                                ],
                                borderColor: [
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(255, 99, 132, 1)'
                                ],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error('Error fetching player data:', error);
                });
        });
    }

    // Save prediction functionality
    const savePredictionButtons = document.querySelectorAll('.save-prediction');
    if (savePredictionButtons.length > 0) {
        savePredictionButtons.forEach(button => {
            button.addEventListener('click', function(event) {
                event.preventDefault();
                
                const predictionId = this.dataset.predictionId;
                
                fetch('/save_prediction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
                    },
                    body: JSON.stringify({ prediction_id: predictionId })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.innerHTML = '<i class="fas fa-check"></i> Saved';
                        this.classList.remove('btn-outline-primary');
                        this.classList.add('btn-success');
                        this.disabled = true;
                    } else {
                        alert('Error saving prediction: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error saving prediction:', error);
                    alert('Error saving prediction. Please try again.');
                });
            });
        });
    }
});
