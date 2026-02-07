import { Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/AppShell'
import Dashboard from './pages/Dashboard'
import StacksList from './pages/StacksList'
import StackDetail from './pages/StackDetail'

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/stacks" element={<StacksList />} />
        <Route path="/stacks/:id" element={<StackDetail />} />
        <Route path="/stacks/:id/:tab" element={<StackDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  )
}
