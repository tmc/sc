import { useEffect } from 'preact/hooks';
import { EditorPane } from './components/EditorPane';
import { useMachinesStore } from './store';
import { SAMPLES } from './data/samples';

function App() {
  const { loadSamples } = useMachinesStore();

  useEffect(() => {
    loadSamples(SAMPLES);
  }, []);

  return (
    <EditorPane />
  )
}

export default App
