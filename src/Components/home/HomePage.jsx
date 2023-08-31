import {Link} from 'react-router-dom';
import {ThemeProvider} from '@moneta/moneta-web-react';

export default function HomePage() {
    const{isLightTheme}= useTheme();

    return (
        <main>
             <div className="container-fluid pt-3">
                <div className=className={isLightTheme ?  "bg-light p-5 m-3 border rounded-2" : 'bg-dark p-5 m-3 border rounded-2'>
                        <h1>React Demo</h1>
                        <p><Link to="/about" className="btn btn-primary btn-lg">Learn More</Link></p>
                </div>
            </div>
        </main>
    );

};