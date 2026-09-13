define([], function() {
    'use strict';

    const update = (url) => fetch(url, {credentials: 'same-origin', cache: 'no-store'})
        .then(response => response.ok ? response.json() : Promise.reject(response.status))
        .then(data => {
            const bar = document.getElementById('local-quizanalytics-progress-bar');
            const message = document.getElementById('local-quizanalytics-progress-message');
            if (!bar || !message) {
                return true;
            }
            const percent = Math.max(0, Math.min(100, Number(data.percent) || 0));
            bar.style.width = percent + '%';
            bar.setAttribute('aria-valuenow', percent);
            bar.textContent = percent + '%';
            message.textContent = data.message || '';
            if (data.completed && data.total) {
                message.textContent += ' (' + data.completed + '/' + data.total + ')';
            }
            const details = data.details || {};
            const diagnostics = [];
            if (data.elapsed) {
                diagnostics.push('Elapsed: ' + data.elapsed + 's');
            }
            if (details.current_metric) {
                diagnostics.push('Current metric: ' + details.current_metric +
                    ' (' + details.current_metric_seconds + 's)');
            }
            if (details.last_item) {
                diagnostics.push('Last quiz: ' + details.last_item +
                    (details.last_item_seconds ? ' (' + details.last_item_seconds + 's)' : ''));
            }
            if (details.slowest_item && details.slowest_item.name) {
                diagnostics.push('Slowest: ' + details.slowest_item.name +
                    ' (' + details.slowest_item.seconds + 's)');
            }
            if (details.timings) {
                const timings = details.timings;
                if (timings.response_fetch_seconds !== undefined) {
                    diagnostics.push('Response fetch: ' + timings.response_fetch_seconds + 's');
                }
                if (timings.analytics_seconds !== undefined) {
                    diagnostics.push('Analytics: ' + timings.analytics_seconds + 's');
                }
            }
            if (diagnostics.length) {
                message.textContent += ' — ' + diagnostics.join(' · ');
            }
            if (data.status === 'complete') {
                window.setTimeout(() => window.location.reload(), 5000);
                return true;
            }
            return data.status === 'failed';
        });

    return {
        init: function(url) {
            const poll = () => update(url).catch(() => false).then(done => {
                if (!done) {
                    window.setTimeout(poll, 3000);
                }
            });
            poll();
        }
    };
});
