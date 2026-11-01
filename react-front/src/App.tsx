import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type Cluster = {
  data_index: string
  size: number
  labels: string[]
  score: number
}

type Result = {
  line: string
  data_index: string
  es_score: number
  rerank_score?: number
}

type SearchResponse = {
  query: string
  clusters: Cluster[]
  results: Result[]
  rejected: Result[]
  rerank_model: string | null
  timings: Record<string, number>
}

function ScoreBar({ score }: { score: number }) {
  const level = score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low'
  return (
    <div className="score" title={`Score de rerank : ${score.toFixed(3)}`}>
      <div className="score-track">
        <div className={`score-fill ${level}`} style={{ width: `${Math.round(score * 100)}%` }} />
      </div>
      <span className="score-value">{score.toFixed(2)}</span>
    </div>
  )
}

function ResultItem({ result, rank }: { result: Result; rank: number }) {
  return (
    <li className="result">
      <span className="rank">{rank}</span>
      <div className="result-body">
        <p className="result-line">{result.line}</p>
        <div className="result-meta">
          <span className="badge">{result.data_index}</span>
          <span>score hybride ES : {result.es_score.toFixed(2)}</span>
        </div>
      </div>
      {result.rerank_score !== undefined && <ScoreBar score={result.rerank_score} />}
    </li>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [clusters, setClusters] = useState(3)
  const [size, setSize] = useState(10)
  const [useRerank, setUseRerank] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSearch(event: FormEvent) {
    event.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, clusters, size, rerank: useRerank }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail ?? `Erreur ${response.status}`)
      setData(body)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <header className="header">
        <h1>IA Search</h1>
        <p>
          Recherche dans les consultations de marchés publics : clusters par embedding, recherche
          hybride ElasticSearch, tri par un modèle rerank.
        </p>
      </header>

      <form className="search" onSubmit={handleSearch}>
        <div className="search-row">
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="ex. rénovation énergétique de logements"
            aria-label="Recherche"
            autoFocus
          />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Recherche…' : 'Search'}
          </button>
        </div>
        <div className="options">
          <label>
            Clusters
            <input type="number" min={1} max={20} value={clusters} onChange={(e) => setClusters(Number(e.target.value))} />
          </label>
          <label>
            Résultats
            <input type="number" min={1} max={50} value={size} onChange={(e) => setSize(Number(e.target.value))} />
          </label>
          <label className="checkbox">
            <input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} />
            Rerank par LLM
          </label>
        </div>
      </form>

      {error && <p className="error">{error}</p>}

      {data && (
        <section className="response">
          <div className="clusters">
            <h2>Clusters les plus proches</h2>
            <ul>
              {data.clusters.map((cluster) => (
                <li key={cluster.data_index} title={cluster.labels.join('\n')}>
                  <span className="badge">{cluster.data_index}</span>
                  <span className="cluster-info">
                    {cluster.size} docs · similarité {cluster.score.toFixed(3)}
                  </span>
                  <span className="cluster-label">{cluster.labels[0]}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="results">
            <h2>
              {data.results.length} résultat{data.results.length > 1 ? 's' : ''}
              <span className="timings">
                {Object.entries(data.timings)
                  .map(([step, seconds]) => `${step} ${seconds.toFixed(2)}s`)
                  .join(' · ')}
              </span>
            </h2>
            {data.results.length === 0 && (
              <p className="empty">
                Aucun document pertinent : la question n'a probablement pas de rapport avec ce corpus.
              </p>
            )}
            <ol>
              {data.results.map((result, index) => (
                <ResultItem key={`${result.data_index}-${index}`} result={result} rank={index + 1} />
              ))}
            </ol>
          </div>

          {data.rejected.length > 0 && (
            <details className="rejected">
              <summary>{data.rejected.length} résultat(s) écarté(s) par le rerank (score &lt; 0.4)</summary>
              <ol>
                {data.rejected.map((result, index) => (
                  <ResultItem key={`${result.data_index}-${index}`} result={result} rank={index + 1} />
                ))}
              </ol>
            </details>
          )}
        </section>
      )}
    </main>
  )
}

export default App
