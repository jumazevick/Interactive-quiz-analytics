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
            if (data.status === 'complete') {
                window.setTimeout(() => window.location.reload(), 500);
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
