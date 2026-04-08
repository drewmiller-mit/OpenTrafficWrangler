import axios from 'axios';

export const createJob = (payload) =>
  axios.post('/api/jobs', payload).then(res => res.data);

export const fetchJob = (id) =>
  axios.get(`/api/jobs/${id}`).then(res => res.data);

export const downloadFile = (id, artifact) =>
  axios.get(`/api/jobs/${id}/files/${artifact}`, { responseType: 'blob' });

export const snapCoordinate = payload =>
  axios.post('/api/snap', payload).then(res => res.data);

export const listIntermediates = () =>
  axios.get('/api/intermediates').then(res => res.data);

export const validateNetwork = payload =>
  axios.post('/api/validate-network', payload).then(res => res.data);

export const validateNetworkOsm = payload =>
  axios.post('/api/validate-network/osm', payload).then(res => res.data);

export const clearIntermediates = () =>
  axios.delete('/api/intermediates').then(res => res.data);

export const fetchOsmFeatures = params =>
  axios
    .get('/api/osm/features', {
      params
    })
    .then(res => res.data);
