document.addEventListener('DOMContentLoaded', function() {
    let dataset = [];
    let centroids = [];
    let prevCentroids = []; // Store previous centroids to detect convergence
    let manualCentroids = [];
    let k = 3; // Default number of clusters
    let clusterAssignments = [];
    let clusterColors = [];
    let converged = false; // Track if KMeans has converged

    // Get buttons for Step through KMeans and Run to Convergence
    const stepBtn = document.getElementById('step-btn');
    const convergeBtn = document.getElementById('converge-btn');

    // Disable buttons based on initialization method
    function checkInitializationMethod() {
        const method = document.getElementById('init-method').value;
        if (method === 'manual') {
            // Disable buttons and reset manual centroids for manual initialization
            stepBtn.disabled = true;
            convergeBtn.disabled = true;
            manualCentroids = [];
            plotData(dataset, manualCentroids, clusterAssignments);
        } else {
            stepBtn.disabled = false;
            convergeBtn.disabled = false;
        }
    }

    // Call checkInitializationMethod when initialization method changes
    document.getElementById('init-method').addEventListener('change', function() {
        checkInitializationMethod();
    });

    // Update k when the user changes the number of clusters
    document.getElementById('clusters').addEventListener('change', function() {
        k = document.getElementById('clusters').value;
        manualCentroids = [];
        plotData(dataset, manualCentroids, clusterAssignments);

        // Reset buttons if in manual mode
        const method = document.getElementById('init-method').value;
        if (method === 'manual') {
            stepBtn.disabled = true;
            convergeBtn.disabled = true;
        }
    });

    // Helper function to check if centroids have converged
    function haveCentroidsConverged(currentCentroids, previousCentroids) {
        if (currentCentroids.length !== previousCentroids.length) return false;
        for (let i = 0; i < currentCentroids.length; i++) {
            if (currentCentroids[i][0] !== previousCentroids[i][0] || currentCentroids[i][1] !== previousCentroids[i][1]) {
                return false;
            }
        }
        return true;
    }

    // Helper function to constrain centroids to the display range [-10, 10]
    function constrainToRange(centroids) {
        return centroids.map(([x, y]) => {
            x = Math.max(-10, Math.min(10, x));  // Constrain x to [-10, 10]
            y = Math.max(-10, Math.min(10, y));  // Constrain y to [-10, 10]
            return [x, y];
        });
    }

    // Generate a new dataset
    document.getElementById('generate-btn').addEventListener('click', function() {
        fetch('/generate_dataset')
            .then(response => response.json())
            .then(data => {
                dataset = data.data;
                centroids = [];
                manualCentroids = [];
                clusterAssignments = [];
                clusterColors = []; // Reset colors for new dataset
                prevCentroids = [];  // Reset previous centroids
                converged = false;   // Reset convergence status
                plotData(dataset, centroids, clusterAssignments);

                // Check initialization method and disable buttons if needed
                checkInitializationMethod();
            });
    });

    // Step through KMeans and initialize centroids
    stepBtn.addEventListener('click', function() {
        if (converged) {
            alert("KMeans has already converged.");
            return;
        }

        let method = document.getElementById('init-method').value;

        if (centroids.length === 0) {  // Initialize centroids if not done yet
            fetch('/initialize_centroids', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({data: dataset, k: k, method: method, manual_centroids: manualCentroids})
            })
            .then(response => response.json())
            .then(data => {
                centroids = constrainToRange(data.centroids); // Constrain centroids to display range
                plotData(dataset, centroids, clusterAssignments);  // Plot initialized centroids
            });
        } else {
            // Step through KMeans
            fetch('/step_kmeans', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({data: dataset, centroids: centroids})
            })
            .then(response => response.json())
            .then(data => {
                prevCentroids = [...centroids]; // Store current centroids before update
                centroids = data.new_centroids;
                clusterAssignments = data.clusters;

                // Assign fixed colors for each cluster (only the first time clusters are formed)
                if (clusterColors.length === 0) {
                    clusterColors = generateClusterColors(k);
                }

                plotData(dataset, centroids, clusterAssignments);

                // Check for convergence by comparing centroids
                if (haveCentroidsConverged(centroids, prevCentroids)) {
                    alert("KMeans has converged.");
                    converged = true;
                }
            });
        }
    });

    // Run to Convergence
    convergeBtn.addEventListener('click', function() {
        fetch('/run_to_convergence', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({data: dataset, centroids: centroids})
        })
        .then(response => response.json())
        .then(data => {
            prevCentroids = [...centroids]; // Store current centroids before update
            centroids = data.centroids;

            // Assign fixed colors for each cluster (only the first time clusters are formed)
            if (clusterColors.length === 0) {
                clusterColors = generateClusterColors(k);
            }

            plotData(dataset, centroids, clusterAssignments);

            // Check for convergence after running to convergence
            if (haveCentroidsConverged(centroids, prevCentroids)) {
                alert("KMeans has converged.");
                converged = true;
            }
        });
    });

    // Reset the algorithm
    document.getElementById('reset-btn').addEventListener('click', function() {
        centroids = [];
        manualCentroids = [];
        clusterAssignments = [];
        clusterColors = []; // Reset colors for reset
        prevCentroids = [];  // Reset previous centroids
        converged = false;   // Reset convergence status
        plotData(dataset, centroids, clusterAssignments);

        // Disable buttons after reset if manual method is selected
        checkInitializationMethod();
    });

    // Initialize the plot first using Plotly.newPlot to attach the plotly_click event
    Plotly.newPlot('plot', [], {
        title: 'KMeans Clustering Data',
        xaxis: { range: [-10, 10] },
        yaxis: { range: [-10, 10] }
    });

    // Handle manual centroid selection using Plotly's plotly_click event
    var plotDiv = document.getElementById('plot');
    plotDiv.on('plotly_click', function(eventData) {
        let method = document.getElementById('init-method').value;
        if (method === 'manual' && manualCentroids.length < k) {
            // Get the click coordinates from the event
            let x = eventData.points[0].x;
            let y = eventData.points[0].y;

            // Add the selected point as a centroid
            manualCentroids.push([x, y]);

            // Update the plot with selected centroids
            plotData(dataset, manualCentroids, clusterAssignments);

            // Enable buttons when enough centroids are selected
            if (manualCentroids.length >= k) {
                stepBtn.disabled = false;
                convergeBtn.disabled = false;
                alert("You have selected all required centroids.");
            }
        }
    });

    // Function to plot the data points and centroids, color data points based on clusters
    function plotData(data, centroids, clusters) {
        let trace1 = [];

        // If clusters are assigned, color data points based on cluster index
        if (clusters.length > 0) {
            clusters.forEach((cluster, clusterIndex) => {
                trace1.push({
                    x: cluster.map(point => point[0]),
                    y: cluster.map(point => point[1]),
                    mode: 'markers',
                    type: 'scatter',
                    marker: { color: clusterColors[clusterIndex], size: 6 },
                    name: `Cluster ${clusterIndex + 1}`
                });
            });
        } else {
            // If no clusters, plot data points without clusters
            trace1.push({
                x: data.map(point => point[0]),
                y: data.map(point => point[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'Data Points'
            });
        }

        // Centroids plot
        let trace2 = {
            x: centroids.map(c => c[0]),
            y: centroids.map(c => c[1]),
            mode: 'markers',
            marker: { size: 12, color: 'red' },
            type: 'scatter',
            name: 'Centroids'
        };

        let layout = {
            title: 'KMeans Clustering Data',
            showlegend: true,
            legend: {"orientation": "h"},
            xaxis: { range: [-10, 10] },  // Constrain x-axis range
            yaxis: { range: [-10, 10] }   // Constrain y-axis range
        };

        let dataPlot = [...trace1, trace2];
        Plotly.react('plot', dataPlot, layout);  // Use Plotly.react for updating without resetting
    }

    // Function to generate fixed distinct colors for clusters
    function generateClusterColors(k) {
        const colors = [];
        for (let i = 0; i < k; i++) {
            const hue = (i * 360 / k) % 360;
            colors.push(`hsl(${hue}, 100%, 50%)`);
        }
        return colors;
    }
});
