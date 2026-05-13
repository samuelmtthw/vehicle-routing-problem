import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Navbar } from './components/ui/Navbar'
import Dashboard from './pages/Dashboard'
import Scenarios from './pages/Scenarios'
import ScenarioDetail from './pages/ScenarioDetail'

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display:'flex', flexDirection:'column', height:'100vh', overflow:'hidden' }}>
        <Navbar />
        <main style={{ flex:1, marginTop:56, overflowY:'auto' }}>
          <Routes>
            <Route path="/"                        element={<Dashboard />} />
            <Route path="/scenarios"               element={<Scenarios />} />
            <Route path="/scenarios/:id"           element={<ScenarioDetail />} />
            <Route path="*"                        element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
