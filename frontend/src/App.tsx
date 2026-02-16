import { Routes, Route, Navigate } from 'react-router-dom'
import AppShell from './components/AppShell'
import Landing from './pages/Landing'
import StacksList from './pages/StacksList'
import StackDetail from './pages/StackDetail'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<StacksList />} />
        <Route path="/stacks" element={<StacksList />} />
        <Route path="/stacks/:id" element={<StackDetail />} />
        <Route path="/stacks/:id/:tab" element={<StackDetail />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
