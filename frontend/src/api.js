const API_BASE = window.location.protocol === 'file:' ? 'http://127.0.0.1:8787' : '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    signal: options.signal,
    ...options,
  });

  const contentType = response.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const error = new Error(payload?.message || payload?.error || `Request failed with status ${response.status}`);
    error.payload = payload;
    throw error;
  }

  return payload;
}

export function getBootstrap() {
  return request('/api/bootstrap');
}

export function listInvestigations() {
  return request('/api/investigations');
}

export function getInvestigation(id, options = {}) {
  return request(`/api/investigations/${encodeURIComponent(id)}`, options);
}

export function getWorkspaceInvestigation(id, options = {}) {
  return request(`/api/investigations/${encodeURIComponent(id)}/workspace`, options);
}

export function runInvestigation(payload) {
  return request('/api/investigations', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function cancelInvestigation(id) {
  return request(`/api/investigations/${encodeURIComponent(id)}/cancel`, {
    method: 'POST',
  });
}

export function isApiLocal() {
  return API_BASE === '';
}
