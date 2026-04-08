import { useEffect, useRef, useState } from 'react'
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery
} from '@tanstack/react-query'
import L from 'leaflet'
import {
  createJob,
  downloadFile,
  fetchJob,
  listIntermediates,
  snapCoordinate,
  validateNetwork,
  validateNetworkOsm,
  clearIntermediates,
  fetchOsmFeatures
} from './api'
import 'leaflet/dist/leaflet.css'
import './App.css'

const queryClient = new QueryClient()
const MIN_INSPECT_ZOOM = 11

const DEFAULT_FORM = {
  interstate: '',
  out_lanes_csv: 'lanes.csv',
  out_ramps_csv: 'ramps.csv',
  anchor_postmile: '0',
  end_postmile: '',
  stationing_direction: 'ascending',
  bbox_buffer_ft: '0.08',
  path_mode: 'normal',
  ref_list: ''
}

const DEFAULT_CENTER = [37.5, -96.5]

const buildMarkerIcon = variant =>
  L.divIcon({
    className: `selection-marker ${variant}`,
    html: '<span></span>',
    iconSize: [22, 22],
    iconAnchor: [11, 11]
  })

const START_ICON = buildMarkerIcon('start')
const END_ICON = buildMarkerIcon('end')
const EMPTY_MANUAL_COORD = { lat: '', lng: '' }
const EMPTY_POINT_STATE = { markerCoord: null, snappedCoord: null }

const formatCoord = value => Number(value).toFixed(6)

function InfoTooltip({ text }) {
  return (
    <span className="info-icon" data-tooltip={text} role="img" aria-label="Info">
      ⓘ
    </span>
  )
}

function SelectionMap({
  startCoord,
  endCoord,
  activePoint,
  onSelect,
  fullscreen = false,
  onToggleFullscreen = () => {},
  inspectionEnabled = false,
  onFeatureSelect = () => {},
  onFeatureError = () => {},
  onFeatureLoadingChange = () => {},
  instructionPrimary = '',
  instructionSecondary = '',
  snapMessage = '',
  selectedFeature = null,
  featureError = null,
  featureLoading = false,
  onToggleInspect = () => {}
}) {
  const containerRef = useRef(null)
  const mapRef = useRef(null)
  const markersRef = useRef({ start: null, end: null })
  const activePointRef = useRef(activePoint)
  const onSelectRef = useRef(onSelect)
  const inspectionRef = useRef(inspectionEnabled)
  const [hoverCoord, setHoverCoord] = useState(null)
  const [ready, setReady] = useState(false)
  const featureLayerRef = useRef(null)
  const highlightedLayerRef = useRef(null)
  const featureRequestIdRef = useRef(0)

  useEffect(() => {
    activePointRef.current = activePoint
  }, [activePoint])

  useEffect(() => {
    onSelectRef.current = onSelect
  }, [onSelect])

  useEffect(() => {
    inspectionRef.current = inspectionEnabled
  }, [inspectionEnabled])

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = L.map(containerRef.current, {
      center: DEFAULT_CENTER,
      zoom: 5,
      zoomControl: true
    })

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map)

    map.on('click', e => {
      if (inspectionRef.current) return
      const target = activePointRef.current
      if (!target) return
      onSelectRef.current(target, e.latlng)
    })

    map.on('mousemove', e => {
      setHoverCoord({
        lat: Number(e.latlng.lat.toFixed(5)),
        lng: Number(e.latlng.lng.toFixed(5))
      })
    })

    map.on('mouseout', () => setHoverCoord(null))

    map.whenReady(() => setReady(true))

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    if (mapRef.current) {
      setTimeout(() => {
        mapRef.current?.invalidateSize()
      }, 100)
    }
  }, [fullscreen])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const { start, end } = markersRef.current

    if (startCoord) {
      if (!start) {
        markersRef.current.start = L.marker([startCoord.lat, startCoord.lng], { icon: START_ICON }).addTo(map)
      } else {
        start.setLatLng([startCoord.lat, startCoord.lng])
      }
    } else if (start) {
      map.removeLayer(start)
      markersRef.current.start = null
    }

    if (endCoord) {
      if (!end) {
        markersRef.current.end = L.marker([endCoord.lat, endCoord.lng], { icon: END_ICON }).addTo(map)
      } else {
        end.setLatLng([endCoord.lat, endCoord.lng])
      }
    } else if (end) {
      map.removeLayer(end)
      markersRef.current.end = null
    }
  }, [startCoord, endCoord])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (startCoord && endCoord) {
      const bounds = L.latLngBounds(
        L.latLng(startCoord.lat, startCoord.lng),
        L.latLng(endCoord.lat, endCoord.lng)
      ).pad(0.15)
      map.flyToBounds(bounds, { duration: 0.8 })
    } else if (startCoord) {
      map.flyTo([startCoord.lat, startCoord.lng], Math.max(map.getZoom(), 12), { duration: 0.6 })
    } else if (endCoord) {
      map.flyTo([endCoord.lat, endCoord.lng], Math.max(map.getZoom(), 12), { duration: 0.6 })
    }
  }, [startCoord, endCoord])

  const clearFeatureLayers = () => {
    if (mapRef.current && featureLayerRef.current) {
      mapRef.current.removeLayer(featureLayerRef.current)
    }
    featureLayerRef.current = null
    highlightedLayerRef.current = null
  }

  useEffect(() => {
    if (!inspectionEnabled) {
      clearFeatureLayers()
      onFeatureSelect(null)
      onFeatureError(null)
      onFeatureLoadingChange(false)
      return
    }
    const map = mapRef.current
    if (!map) return

    const fetchFeatures = () => {
      if (!mapRef.current || !inspectionEnabled) {
        return
      }
      const zoom = mapRef.current.getZoom()
      if (zoom < MIN_INSPECT_ZOOM) {
        clearFeatureLayers()
        onFeatureSelect(null)
        onFeatureLoadingChange(false)
        onFeatureError(`Zoom in (≥${MIN_INSPECT_ZOOM}) to inspect OSM details.`)
        return
      }
      const bounds = mapRef.current.getBounds()
      const params = {
        south: bounds.getSouth(),
        west: bounds.getWest(),
        north: bounds.getNorth(),
        east: bounds.getEast()
      }
      featureRequestIdRef.current += 1
      const requestId = featureRequestIdRef.current
      onFeatureLoadingChange(true)
      onFeatureError(null)
      fetchOsmFeatures(params)
        .then(data => {
          if (requestId !== featureRequestIdRef.current || !inspectionEnabled) {
            return
          }
          onFeatureLoadingChange(false)
          clearFeatureLayers()
          if (!data?.features?.length) {
            onFeatureSelect(null)
            return
          }
          const layer = L.geoJSON(data, {
            style: () => ({
              color: '#0f9d58',
              weight: 3,
              opacity: 0.9
            }),
            onEachFeature: (feature, layerItem) => {
              layerItem.on('click', () => {
                if (highlightedLayerRef.current) {
                  highlightedLayerRef.current.setStyle({ color: '#0f9d58', weight: 3 })
                }
                layerItem.setStyle({ color: '#ec4899', weight: 4 })
                highlightedLayerRef.current = layerItem
                const properties = feature.properties || {}
                const { id: osmId, ...tags } = properties
                onFeatureSelect({
                  id: osmId || feature.id,
                  tags,
                  geometryType: feature.geometry?.type || 'LineString'
                })
              })
            }
          })
          featureLayerRef.current = layer
          layer.addTo(mapRef.current)
        })
        .catch(error => {
          if (requestId !== featureRequestIdRef.current || !inspectionEnabled) {
            return
          }
          onFeatureLoadingChange(false)
          onFeatureError(error?.response?.data?.detail || 'Unable to load OSM features.')
        })
    }

    fetchFeatures()
    map.on('moveend', fetchFeatures)
    return () => {
      map.off('moveend', fetchFeatures)
      clearFeatureLayers()
    }
  }, [inspectionEnabled, onFeatureSelect, onFeatureError, onFeatureLoadingChange])

  useEffect(() => {
    if (fullscreen) {
      document.body.classList.add('map-fullscreen-active')
    } else {
      document.body.classList.remove('map-fullscreen-active')
    }
    return () => {
      document.body.classList.remove('map-fullscreen-active')
    }
  }, [fullscreen])

  const mapClasses = ['leaflet-map']
  if (fullscreen) mapClasses.push('fullscreen')
  if (inspectionEnabled) mapClasses.push('inspecting')

  return (
    <div className={mapClasses.join(' ')}>
      {!ready && <div className="map-placeholder"><p>Loading map...</p></div>}
      {hoverCoord && (
        <div className="cursor-coords">
          <span>Lat: {hoverCoord.lat}</span>
          <span>Lon: {hoverCoord.lng}</span>
        </div>
      )}
      {fullscreen && (instructionPrimary || instructionSecondary) && (
        <div className="map-instructions">
          {instructionPrimary && <p className="map-instructions__primary">{instructionPrimary}</p>}
          {instructionSecondary && <p className="map-instructions__secondary">{instructionSecondary}</p>}
          {snapMessage && <p className="map-instructions__snap">{snapMessage}</p>}
        </div>
      )}
      <button
        type="button"
        className="map-fullscreen-toggle"
        onClick={() => onToggleFullscreen(!fullscreen)}
        aria-label={fullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
      >
        {fullscreen ? '✕' : '⛶'}
      </button>
      {fullscreen && (
        <div className="map-fullscreen-controls">
          <button
            type="button"
            className={`point-btn ${inspectionEnabled ? 'active' : ''}`}
            onClick={onToggleInspect}
          >
            {inspectionEnabled ? 'Exit inspect mode' : 'Inspect OSM'}
          </button>
          {!inspectionEnabled && (
            <p className="muted small">Tap to place {activePoint === 'start' ? 'start' : 'end'} point.</p>
          )}
        </div>
      )}
      {fullscreen && inspectionEnabled && (
        <div className="map-inspector-overlay">
          {featureError ? (
            <p className="error">{featureError}</p>
          ) : featureLoading ? (
            <p className="muted">Loading OSM features...</p>
          ) : selectedFeature ? (
            <>
              <p className="map-inspector-overlay__title">
                Way ID{' '}
                {selectedFeature.id ? (
                  <a
                    href={`https://www.openstreetmap.org/way/${selectedFeature.id}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {selectedFeature.id}
                  </a>
                ) : (
                  'Unknown'
                )}
              </p>
              <p className="muted small">Geometry: {selectedFeature.geometryType || 'LineString'}</p>
              <div className="map-inspector-overlay__tags">
                {Object.keys(selectedFeature.tags || {}).length === 0 ? (
                  <p className="muted">No tags available.</p>
                ) : (
                  Object.entries(selectedFeature.tags).map(([key, value]) => (
                    <p key={key}>
                      <strong>{key}:</strong> {String(value)}
                    </p>
                  ))
                )}
              </div>
            </>
          ) : (
            <p className="muted">Click a highlighted way to view its metadata.</p>
          )}
        </div>
      )}
    <div ref={containerRef} className="map-canvas" />
  </div>
)
}

function Dashboard() {
  const [formValues, setFormValues] = useState(DEFAULT_FORM)
  const [jobId, setJobId] = useState(null)
  const [startPoint, setStartPoint] = useState(EMPTY_POINT_STATE)
  const [endPoint, setEndPoint] = useState(EMPTY_POINT_STATE)
  const [startManualCoord, setStartManualCoord] = useState(EMPTY_MANUAL_COORD)
  const [endManualCoord, setEndManualCoord] = useState(EMPTY_MANUAL_COORD)
  const [activePoint, setActivePoint] = useState('start')
  const [formError, setFormError] = useState(null)
  const [snapStatusMessage, setSnapStatusMessage] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [isMapFullscreen, setIsMapFullscreen] = useState(false)
  const [inspectionEnabled, setInspectionEnabled] = useState(false)
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [featureError, setFeatureError] = useState(null)
  const [featureLoading, setFeatureLoading] = useState(false)
  const lastSelectionRef = useRef('start')
  const startCoord = startPoint.markerCoord
  const endCoord = endPoint.markerCoord
  const startSnappedCoord = startPoint.snappedCoord
  const endSnappedCoord = endPoint.snappedCoord

  const setPointState = (target, updater) => {
    const apply = prev => (typeof updater === 'function' ? updater(prev) : updater)
    if (target === 'start') {
      setStartPoint(apply)
    } else {
      setEndPoint(apply)
    }
  }

  const pointLabel = target => (target === 'start' ? 'Start' : 'End')
  const pointStatus = point => {
    if (point.snappedCoord) return 'Snapped to roadway'
    if (point.markerCoord) return 'Placed on map, not snapped'
    return 'No point set'
  }

  const createJobMutation = useMutation({
    mutationFn: createJob,
    onSuccess: data => {
      setJobId(data.job_id)
    }
  })

  const snapMutation = useMutation({
    mutationFn: ({ latlng, interstate, buffer }) =>
      snapCoordinate({
        interstate,
        lat: latlng.lat,
        lon: latlng.lng,
        bbox_buffer_deg: buffer
      }),
    onSuccess: (data, variables) => {
      const coord = {
        lat: Number(data.lat.toFixed(6)),
        lng: Number(data.lon.toFixed(6))
      }

      setPointState(variables.target, prev => ({
        ...prev,
        markerCoord: coord,
        snappedCoord: coord
      }))
      setSnapStatusMessage(
        `${pointLabel(variables.target)} point snapped to ${formatCoord(coord.lat)}, ${formatCoord(coord.lng)}.`
      )
      setFormError(null)
    },
    onError: error => {
      setSnapStatusMessage('')
      setFormError(error?.response?.data?.detail || 'Unable to snap point to roadway.')
    }
  })

  const validateMutation = useMutation({
    mutationFn: () =>
      validateNetwork({
        interstate: formValues.interstate,
        lanes_filename: formValues.out_lanes_csv,
        ramps_filename: formValues.out_ramps_csv
      }),
    onSuccess: data => {
      const url = `/api/artifacts/file?path=${encodeURIComponent(data.artifact_path)}`
      window.open(url, '_blank', 'noopener')
    },
    onError: error => {
      setFormError(error?.response?.data?.detail || 'Validation failed.')
    }
  })

  const validateOsmMutation = useMutation({
    mutationFn: () =>
      validateNetworkOsm({
        interstate: formValues.interstate,
        lanes_filename: formValues.out_lanes_csv,
        ramps_filename: formValues.out_ramps_csv
      }),
    onSuccess: data => {
      const url = `/api/artifacts/file?path=${encodeURIComponent(data.artifact_path)}&inline=true`
      window.open(url, '_blank', 'noopener')
    },
    onError: error => {
      setFormError(error?.response?.data?.detail || 'OSM validation failed.')
    }
  })

  const setActiveSelection = point => {
    lastSelectionRef.current = point
    setActivePoint(point)
  }

  const handleSelectPointMode = point => {
    if (inspectionEnabled) {
      setInspectionEnabled(false)
    }
    setSelectedFeature(null)
    setFeatureError(null)
    setFeatureLoading(false)
    setActiveSelection(point)
  }

  const toggleInspection = () => {
    if (inspectionEnabled) {
      setInspectionEnabled(false)
      setActiveSelection(lastSelectionRef.current || 'start')
    } else {
      lastSelectionRef.current = activePoint || lastSelectionRef.current || 'start'
      setActivePoint(null)
      setInspectionEnabled(true)
    }
    setSelectedFeature(null)
    setFeatureError(null)
  }

  const handleChange = event => {
    const { name, value, type, checked } = event.target
    setFormValues(prev => ({ ...prev, [name]: type === 'checkbox' ? checked : value }))
  }

  const handlePointSelect = (target, latlng) => {
    if (snapMutation.isPending) return
    setFormError(null)
    setSnapStatusMessage('')
    const coord = {
      lat: Number(latlng.lat.toFixed(6)),
      lng: Number(latlng.lng.toFixed(6))
    }
    setPointState(target, prev => ({
      ...prev,
      markerCoord: coord,
      snappedCoord: null
    }))
    setSnapStatusMessage(
      `${pointLabel(target)} point placed on the map at ${formatCoord(coord.lat)}, ${formatCoord(coord.lng)}. Use "Snap to roadway" when ready.`
    )
  }

  const parseManualCoordinate = (value, type, label) => {
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) {
      throw new Error(`${label} must be a valid number.`)
    }
    if (type === 'lat' && (parsed < -90 || parsed > 90)) {
      throw new Error(`${label} must be between -90 and 90.`)
    }
    if (type === 'lng' && (parsed < -180 || parsed > 180)) {
      throw new Error(`${label} must be between -180 and 180.`)
    }
    return parsed
  }

  const handleManualCoordChange = (target, field, value) => {
    if (target === 'start') {
      setStartManualCoord(prev => ({ ...prev, [field]: value }))
    } else {
      setEndManualCoord(prev => ({ ...prev, [field]: value }))
    }
    setPointState(target, prev => ({ ...prev, snappedCoord: null }))
    setSnapStatusMessage('')
  }

  const parseManualTarget = target => {
    const raw = target === 'start' ? startManualCoord : endManualCoord
    const lat = parseManualCoordinate(raw.lat, 'lat', `${pointLabel(target)} latitude`)
    const lng = parseManualCoordinate(raw.lng, 'lng', `${pointLabel(target)} longitude`)
    return { lat, lng }
  }

  const handleShowOnMap = target => {
    try {
      const coord = parseManualTarget(target)
      setFormError(null)
      setActiveSelection(target)
      setSelectedFeature(null)
      setFeatureError(null)
      setFeatureLoading(false)
      setPointState(target, prev => ({
        ...prev,
        markerCoord: coord,
        snappedCoord: null
      }))
      setSnapStatusMessage(
        `${pointLabel(target)} coordinates shown on the map at ${formatCoord(coord.lat)}, ${formatCoord(coord.lng)}.`
      )
      if (inspectionEnabled) {
        setInspectionEnabled(false)
      }
    } catch (error) {
      setFormError(error.message || 'Invalid manual coordinate input.')
    }
  }

  const handleSnapToRoadway = target => {
    if (snapMutation.isPending) return
    try {
      const coord =
        (target === 'start' ? startPoint.markerCoord : endPoint.markerCoord) || parseManualTarget(target)
      setFormError(null)
      setSnapStatusMessage(`Snapping ${pointLabel(target).toLowerCase()} point to the nearest roadway...`)
      snapMutation.mutate({
        target,
        latlng: coord,
        interstate: formValues.interstate || '',
        buffer: Number(formValues.bbox_buffer_ft || 0.05),
        source: 'point'
      })
    } catch (error) {
      setFormError(error.message || 'Invalid manual coordinate input.')
    }
  }

  const clearPoint = target => {
    setPointState(target, EMPTY_POINT_STATE)
    if (target === 'start') {
      setStartManualCoord(EMPTY_MANUAL_COORD)
    } else {
      setEndManualCoord(EMPTY_MANUAL_COORD)
    }
    setFormError(null)
    setSnapStatusMessage('')
    if (!inspectionEnabled) {
      setActiveSelection(target)
    } else {
      lastSelectionRef.current = target
    }
  }

  const handleSubmit = event => {
    event.preventDefault()

    if (!startSnappedCoord || !endSnappedCoord) {
      setFormError('Set and snap both start and end points (from map or manual coordinates) before submitting.')
      return
    }

    const refList =
      formValues.path_mode === 'normal'
        ? null
        : formValues.ref_list
            ? formValues.ref_list
                .split(',')
                .map(item => item.trim())
                .filter(Boolean)
            : null

    const payload = {
      interstate: formValues.interstate,
      seg_start_lat: startSnappedCoord.lat,
      seg_start_lon: startSnappedCoord.lng,
      seg_end_lat: endSnappedCoord.lat,
      seg_end_lon: endSnappedCoord.lng,
      out_lanes_csv: formValues.out_lanes_csv,
      out_ramps_csv: formValues.out_ramps_csv,
      anchor_postmile: Number(formValues.anchor_postmile || 0),
      end_postmile:
        formValues.end_postmile === '' || formValues.end_postmile === null
          ? null
          : Number(formValues.end_postmile),
      stationing_direction: formValues.stationing_direction,
      bbox_buffer_ft: Number(formValues.bbox_buffer_ft || 0),
      path_mode: formValues.path_mode,
      ref_list: refList
    }

    setJobId(null)
    createJobMutation.mutate(payload)
  }

  const handleReset = () => {
    setFormValues(DEFAULT_FORM)
    setInspectionEnabled(false)
    setSelectedFeature(null)
    setFeatureError(null)
    setFeatureLoading(false)
    setStartPoint(EMPTY_POINT_STATE)
    setEndPoint(EMPTY_POINT_STATE)
    setStartManualCoord(EMPTY_MANUAL_COORD)
    setEndManualCoord(EMPTY_MANUAL_COORD)
    setFormError(null)
    setSnapStatusMessage('')
    setActiveSelection('start')
    setJobId(null)
    createJobMutation.reset()
    queryClient.removeQueries({ queryKey: ['job'] })
  }

  useEffect(() => {
    if (!inspectionEnabled) {
      setSelectedFeature(null)
      setFeatureError(null)
      setFeatureLoading(false)
    }
  }, [inspectionEnabled])

  useEffect(() => {
    if (startPoint.markerCoord) {
      setStartManualCoord({
        lat: formatCoord(startPoint.markerCoord.lat),
        lng: formatCoord(startPoint.markerCoord.lng)
      })
    }
  }, [startPoint.markerCoord])

  useEffect(() => {
    if (endPoint.markerCoord) {
      setEndManualCoord({
        lat: formatCoord(endPoint.markerCoord.lat),
        lng: formatCoord(endPoint.markerCoord.lng)
      })
    }
  }, [endPoint.markerCoord])

  const activeSelectionLabel =
    (activePoint || lastSelectionRef.current || 'start') === 'start' ? 'Start' : 'End'
  const mapInstructionPrimary = inspectionEnabled
    ? 'Inspect mode enabled: click highlighted OSM ways to view metadata.'
    : 'Click on the map to place exact start and end points, then snap each one to a roadway.'
  const mapInstructionSecondary = inspectionEnabled
    ? 'Exit inspect mode to resume selecting start and end points.'
    : `Active selection: ${activeSelectionLabel}`
  const validationDisabled = !(formValues.interstate || '').trim()
  const kmlErrorMessage = validateMutation.isError
    ? validateMutation.error?.response?.data?.detail || 'Validation failed.'
    : ''
  const osmErrorMessage = validateOsmMutation.isError
    ? validateOsmMutation.error?.response?.data?.detail || 'OSM validation failed.'
    : ''

  return (
    <main>
      <section className="panel">
        <div className="panel-header">
          <h1>Segment Extraction Job</h1>
          <button
            type="button"
            className="settings-toggle"
            onClick={() => setSettingsOpen(prev => !prev)}
          >
            {settingsOpen ? 'Hide settings' : 'Settings'}
          </button>
        </div>
        <p className="muted onboarding-hint">
          Enter your route (e.g. 'I5' or 'CA SR 55'), then place start/end points by clicking the map or by entering coordinates manually. Both points must be snapped to a roadway before submission. Mainline extraction now starts strict and relaxes automatically if needed, so review the job logs whenever the run reports multiple possible paths.
        </p>
        <form onSubmit={handleSubmit} className="job-form">
          <label>
            Interstate
            <input
              name="interstate"
              value={formValues.interstate}
              onChange={handleChange}
              required
            />
          </label>

          <div className="map-section">
            <div className="map-header">
              <div>
                <p className="muted">Set points by clicking the map or entering latitude/longitude manually.</p>
                <p className="muted">
                  Active selection: <strong>{activeSelectionLabel}</strong>
                </p>
                {snapMutation.isPending && (
                  <p className="muted">
                    Snapping to the nearest roadway...
                  </p>
                )}
              </div>
              <div className="map-controls">
                <div className="point-toggle">
                  <button
                    type="button"
                    className={`point-btn ${activePoint === 'start' ? 'active' : ''}`}
                    onClick={() => handleSelectPointMode('start')}
                    disabled={inspectionEnabled}
                  >
                    Set start
                  </button>
                  <button
                    type="button"
                    className={`point-btn ${activePoint === 'end' ? 'active' : ''}`}
                    onClick={() => handleSelectPointMode('end')}
                    disabled={inspectionEnabled}
                  >
                    Set end
                  </button>
                </div>
                <div className="map-tools">
                  <button
                    type="button"
                    className={`point-btn ${inspectionEnabled ? 'active' : ''}`}
                    onClick={toggleInspection}
                  >
                    {inspectionEnabled ? 'Exit inspect mode' : 'Inspect OSM'}
                  </button>
                </div>
                {inspectionEnabled && (
                  <p className="muted small align-right">Inspect mode active — exit to edit start/end points.</p>
                )}
              </div>
            </div>
            <SelectionMap
              startCoord={startCoord}
              endCoord={endCoord}
              activePoint={activePoint}
              onSelect={handlePointSelect}
              fullscreen={isMapFullscreen}
              onToggleFullscreen={setIsMapFullscreen}
              inspectionEnabled={inspectionEnabled}
              onFeatureSelect={setSelectedFeature}
              onFeatureError={setFeatureError}
              onFeatureLoadingChange={setFeatureLoading}
              instructionPrimary={mapInstructionPrimary}
              instructionSecondary={mapInstructionSecondary}
              snapMessage={snapMutation.isPending ? 'Snapping to the nearest roadway...' : ''}
              selectedFeature={selectedFeature}
              featureError={featureError}
              featureLoading={featureLoading}
              onToggleInspect={toggleInspection}
            />
            <div className="coords-grid">
              <div>
                <p className="coord-label">Start</p>
                <p className="muted small">{pointStatus(startPoint)}</p>
                <div className="coord-pair">
                  <input
                    value={startManualCoord.lat}
                    onChange={event => handleManualCoordChange('start', 'lat', event.target.value)}
                    placeholder="Latitude"
                  />
                  <input
                    value={startManualCoord.lng}
                    onChange={event => handleManualCoordChange('start', 'lng', event.target.value)}
                    placeholder="Longitude"
                  />
                </div>
                <div className="point-action-row">
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => handleShowOnMap('start')}
                    disabled={snapMutation.isPending}
                  >
                    Show on map
                  </button>
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => handleSnapToRoadway('start')}
                    disabled={snapMutation.isPending}
                  >
                    Snap to roadway
                  </button>
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => clearPoint('start')}
                    disabled={snapMutation.isPending}
                  >
                    Clear start point
                  </button>
                </div>
              </div>
              <div>
                <p className="coord-label">End</p>
                <p className="muted small">{pointStatus(endPoint)}</p>
                <div className="coord-pair">
                  <input
                    value={endManualCoord.lat}
                    onChange={event => handleManualCoordChange('end', 'lat', event.target.value)}
                    placeholder="Latitude"
                  />
                  <input
                    value={endManualCoord.lng}
                    onChange={event => handleManualCoordChange('end', 'lng', event.target.value)}
                    placeholder="Longitude"
                  />
                </div>
                <div className="point-action-row">
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => handleShowOnMap('end')}
                    disabled={snapMutation.isPending}
                  >
                    Show on map
                  </button>
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => handleSnapToRoadway('end')}
                    disabled={snapMutation.isPending}
                  >
                    Snap to roadway
                  </button>
                  <button
                    type="button"
                    className="btn ghost manual-snap-btn"
                    onClick={() => clearPoint('end')}
                    disabled={snapMutation.isPending}
                  >
                    Clear end point
                  </button>
                </div>
              </div>
            </div>
            {snapStatusMessage && <p className="muted small snap-status">{snapStatusMessage}</p>}
            <div className="feature-inspector">
              {inspectionEnabled ? (
                featureError ? (
                  <p className="error">{featureError}</p>
                ) : featureLoading ? (
                  <p className="muted">Loading OSM features...</p>
                ) : selectedFeature ? (
                  <>
                    <div className="feature-inspector__header">
                      <p className="feature-inspector__title">
                        Way ID{' '}
                        {selectedFeature.id ? (
                          <a
                            href={`https://www.openstreetmap.org/way/${selectedFeature.id}`}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {selectedFeature.id}
                          </a>
                        ) : (
                          'Unknown'
                        )}
                      </p>
                      <p className="muted small">
                        Geometry: {selectedFeature.geometryType || 'LineString'}
                      </p>
                    </div>
                    <div className="feature-tags">
                      {Object.keys(selectedFeature.tags || {}).length === 0 ? (
                        <p className="muted">No tags available.</p>
                      ) : (
                        <dl>
                          {Object.entries(selectedFeature.tags).map(([key, value]) => (
                            <div key={key}>
                              <dt>{key}</dt>
                              <dd>{String(value)}</dd>
                            </div>
                          ))}
                        </dl>
                      )}
                    </div>
                  </>
                ) : (
                  <p className="muted">Click a highlighted way to view its metadata.</p>
                )
              ) : (
                <p className="muted">Enable “Inspect OSM” to explore roadway attributes.</p>
              )}
            </div>
          </div>

          {settingsOpen && (
            <div className="settings-panel">
              <p className="muted">Advanced options</p>
              <p className="muted small">
                The extractor now starts with strict regex-based motorway filtering and relaxes automatically. If the job logs say multiple possible paths were found, review the extracted path and, if needed, resubmit with Path Mode set to Prefer or Avoid plus exact ref strings.
              </p>
              <div className="field-group">
                <label>
                  <span className="label-row">
                    Postmile at Segment Start
                    <InfoTooltip text={`Use when you know the milepost at your anchor.\n• Enter the signed postmile/station value at your start.\n• All downstream points will add/subtract distance from this baseline.`} />
                  </span>
                  <input
                    type="number"
                    step="any"
                    name="anchor_postmile"
                    value={formValues.anchor_postmile}
                    onChange={handleChange}
                  />
                </label>
                <label>
                  <span className="label-row">
                    Postmile at Segment End (Optional)
                    <InfoTooltip text={`Optional two-point calibration.\n• If provided, stationing is linearly interpolated between start and end postmile.\n• This reduces accumulated drift over long corridors.`} />
                  </span>
                  <input
                    type="number"
                    step="any"
                    name="end_postmile"
                    value={formValues.end_postmile}
                    onChange={handleChange}
                    placeholder="Leave blank to use start-anchor method"
                  />
                </label>
                <label>
                  <span className="label-row">
                    Stationing Direction
                    <InfoTooltip text={`Controls how stationing changes along the corridor.\n• Ascending: mileposts increase away from the start.\n• Descending: mileposts decrease toward the end.`} />
                  </span>
                  <select
                    name="stationing_direction"
                    value={formValues.stationing_direction}
                    onChange={handleChange}
                  >
                    <option value="ascending">Ascending</option>
                    <option value="descending">Descending</option>
                  </select>
                </label>
                <label>
                  <span className="label-row">
                    Bounding Box Buffer (degrees)
                    <InfoTooltip text={`Expands the Overpass bounding box.\n• Keep the default for most straight segments.\n• Increase only if curves fall outside the start/end box.\n• Larger values slow queries and may trigger rate limits.`} />
                  </span>
                  <input
                    type="number"
                    step="any"
                    name="bbox_buffer_ft"
                    value={formValues.bbox_buffer_ft}
                    onChange={handleChange}
                  />
                </label>
                <label>
                  <span className="label-row">
                    Path Mode
                    <InfoTooltip text={`Adjusts how ambiguous path candidates are resolved.\n• Normal: use the default regex-based extraction tree only.\n• Prefer: when multiple candidate paths remain, favor the exact ref strings you list.\n• Avoid: when multiple candidate paths remain, steer away from the exact ref strings you list.`} />
                  </span>
                  <select name="path_mode" value={formValues.path_mode} onChange={handleChange}>
                    <option value="normal">Normal</option>
                    <option value="prefer">Prefer</option>
                    <option value="avoid">Avoid</option>
                  </select>
                </label>
              </div>
              {formValues.path_mode !== 'normal' && (
                <label>
                  <span className="label-row">
                    {formValues.path_mode === 'prefer' ? 'Ref IDs to Prefer' : 'Ref IDs to Avoid'}
                    <InfoTooltip text={`Provide comma-separated exact ref strings (e.g., I-5, US-101).\n• These are used when multiple candidate paths remain at a successful extraction step.\n• Prefer chooses paths with the highest prevalence of your listed refs.\n• Avoid chooses paths with the lowest prevalence of your listed refs.`} />
                  </span>
                  <textarea
                    name="ref_list"
                    value={formValues.ref_list}
                    placeholder="e.g., I-5, US-101"
                    onChange={handleChange}
                  />
                </label>
              )}
            </div>
          )}

          <div className="actions">
            <button className="btn" type="submit" disabled={createJobMutation.isPending}>
              {createJobMutation.isPending ? 'Submitting...' : 'Submit job'}
            </button>
            <button
              className="btn secondary"
              type="button"
              onClick={handleReset}
              disabled={createJobMutation.isPending}
            >
              Reset form
            </button>
            {formError && <p className="error">{formError}</p>}
            {createJobMutation.isError && (
              <p className="error">
                {createJobMutation.error?.response?.data?.detail || 'Job submission failed'}
              </p>
            )}
          </div>
        </form>
      </section>

      <section className="panel status-panel">
        <h2>Job Status</h2>
        <JobStatus jobId={jobId} />
      </section>

      <ValidationPanel
        onValidateKml={() => validateMutation.mutate()}
        onValidateOsm={() => validateOsmMutation.mutate()}
        kmlPending={validateMutation.isPending}
        osmPending={validateOsmMutation.isPending}
        kmlError={kmlErrorMessage}
        osmError={osmErrorMessage}
        disabled={validationDisabled}
      />

      <IntermediateOutputs />
    </main>
  )
}

function JobStatus({ jobId }) {
  const [downloadError, setDownloadError] = useState(null)

  const jobQuery = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => fetchJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: query => {
      const status = query.state.data?.status
      return status && status !== 'finished' && status !== 'failed' ? 4000 : false
    }
  })

  if (!jobId) {
    return <p className="empty-state">Submit a job to see status updates.</p>
  }

  if (jobQuery.isLoading) {
    return <p className="empty-state">Fetching job status...</p>
  }

  if (jobQuery.isError) {
    return (
      <p className="empty-state error">
        Failed to load job: {jobQuery.error?.message || 'Unknown error'}
      </p>
    )
  }

  const job = jobQuery.data
  const logs = job.logs ?? []
  const downloadsReady = job.status === 'finished' && job.outputs
  const downloadItems = [
    { key: 'lanes_csv', label: 'lanes CSV' },
    { key: 'ramps_csv', label: 'ramps CSV' }
  ].filter(item => Boolean(job.outputs?.[item.key]))
  const gmnsReady = Boolean(job.outputs?.gmns_archive)

  const handleDownload = async key => {
    if (!job.outputs?.[key]) return
    setDownloadError(null)
    try {
      const response = await downloadFile(jobId, key)
      const filename = job.outputs[key].split('/').pop() || `${key}.dat`
      const url = URL.createObjectURL(response.data)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (error) {
      setDownloadError(error?.response?.data?.detail || 'Download failed')
    }
  }

  const handleView = key => {
    if (!job.outputs?.[key]) return
    const url = `/api/jobs/${jobId}/files/${key}?inline=true`
    window.open(url, '_blank', 'noopener')
  }

  return (
    <div className="status-content">
      <div className="status-header">
        <p className="job-id">
          Job ID: <code>{jobId}</code>
        </p>
        <span className={`badge ${job.status}`}>{job.status}</span>
      </div>

      {job.message && (
        <p className={job.status === 'failed' ? 'error' : 'success'}>{job.message}</p>
      )}

      <div className="status-grid">
        <div>
          <h3>Logs</h3>
          {logs.length === 0 ? (
            <p className="muted">No logs yet.</p>
          ) : (
            <div className="logs">
              <ul>
                {logs.map(entry => (
                  <li key={entry}>{entry}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div>
          <h3>Downloads</h3>
          {downloadsReady ? (
            <div className="downloads">
              {downloadItems.map(item => (
                <div className="download-pair" key={item.key}>
                  <button className="btn ghost" onClick={() => handleView(item.key)}>
                    View {item.label}
                  </button>
                  <button className="btn secondary" onClick={() => handleDownload(item.key)}>
                    Download {item.label}
                  </button>
                </div>
              ))}
              {gmnsReady && (
                <div className="download-pair">
                  <button className="btn secondary" onClick={() => handleDownload('gmns_archive')}>
                    Download GMNS network
                  </button>
                </div>
              )}
            </div>
          ) : (
            <p className="muted">Artifacts available when the job finishes.</p>
          )}
          {downloadError && <p className="error">{downloadError}</p>}
        </div>
      </div>
    </div>
  )
}

function IntermediateOutputs() {
  const { data, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['intermediates'],
    queryFn: () => listIntermediates(),
    refetchInterval: 15000
  })

  const clearMutation = useMutation({
    mutationFn: () => clearIntermediates(),
    onSuccess: () => {
      refetch()
    }
  })

  const artifactUrl = (path, inline = false) =>
    `/api/artifacts/file?path=${encodeURIComponent(path)}${inline ? '&inline=true' : ''}`

  const handleOpen = path => {
    window.open(artifactUrl(path, true), '_blank', 'noopener')
  }

  const handleDownload = path => {
    const link = document.createElement('a')
    link.href = artifactUrl(path, false)
    link.download = path.split('/').pop() || 'artifact.dat'
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Intermediate Outputs</h2>
        <div className="panel-actions">
          <button type="button" className="settings-toggle" onClick={() => refetch()}>
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </button>
          <button
            type="button"
            className="settings-toggle danger"
            onClick={() => clearMutation.mutate()}
            disabled={clearMutation.isPending}
          >
            {clearMutation.isPending ? 'Clearing…' : 'Clear all'}
          </button>
        </div>
      </div>
      {isLoading ? (
        <p className="muted">Loading intermediates…</p>
      ) : isError ? (
        <p className="error">Unable to load intermediate outputs.</p>
      ) : data.length === 0 ? (
        <p className="muted">Intermediates will appear after a job runs.</p>
      ) : (
        <div className="intermediate-list">
          {data.map(item => (
            <div key={item.relative_path} className="artifact-row">
              <div>
                <p className="artifact-name">{item.name}</p>
                <p className="artifact-meta">
                  {item.mime_type || 'file'} · {(item.size_bytes / 1024).toFixed(1)} KB ·{' '}
                  {new Date(item.updated_at).toLocaleString()} · {item.relative_path}
                </p>
              </div>
              <div className="artifact-actions">
                <button type="button" className="btn ghost" onClick={() => handleOpen(item.relative_path)}>
                  Open
                </button>
                <button
                  type="button"
                  className="btn secondary"
                  onClick={() => handleDownload(item.relative_path)}
                >
                  Download
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      {clearMutation.isError && <p className="error">Failed to clear outputs.</p>}
    </section>
  )
}

function ValidationPanel({
  onValidateKml,
  onValidateOsm,
  kmlPending,
  osmPending,
  kmlError,
  osmError,
  disabled
}) {
  return (
    <section className="panel validation-panel">
      <div className="panel-header">
        <h2>Validation</h2>
      </div>
      <div className="validation-options">
        <div className="validation-card">
          <h3>Google Earth KML</h3>
          <p className="muted">
            Generate a .kml overlay of lanes and ramps for review in Google Earth or other desktop GIS tools.
          </p>
          <button type="button" className="btn secondary" onClick={onValidateKml} disabled={kmlPending || disabled}>
            {kmlPending ? 'Generating KML…' : 'Download KML'}
          </button>
          {kmlError && <p className="error">{kmlError}</p>}
        </div>
        <div className="validation-card">
          <h3>OSM Inspector</h3>
          <p className="muted">
            Open an interactive Kepler map in a new tab to compare extracted ramps and lanes against the OpenStreetMap
            basemap.
          </p>
          <button type="button" className="btn ghost" onClick={onValidateOsm} disabled={osmPending || disabled}>
            {osmPending ? 'Launching viewer…' : 'Open OSM Validator'}
          </button>
          {osmError && <p className="error">{osmError}</p>}
        </div>
      </div>
    </section>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Dashboard />
    </QueryClientProvider>
  )
}

export default App
