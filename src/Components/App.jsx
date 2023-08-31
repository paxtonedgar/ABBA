import {Router, Routes, BrowserRouter} from 'react-router-dom';
import {ThemeProvider} from '@moneta/moneta-web-react';

import Header from './Header';
import HomePage from './home/HomePage';
import AboutPage from './about/AboutPage';
import OnboardEvent from './OnboardEvent/OnboardEvent';

function App() {
  return (
    <ThemeProvider>
        <BrowserRouter>
            <div className="fixed-layout">
                <Header />

                <Routes>
                    <Route path="/" element={HomePage /} />
                    <Route path= "/about" element={<AboutPage />} />
                     <Route path= "/OnboardEvent" component={<OnboardEvent />} />
                </Routes>


            </div>
        </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
