import {NavLink} from 'react-router-dom';

export default function AboutPage(){
    return (
    <div>
        <nav classname-"side-menu">
        <ul classname="list-group list-group-nav"></ul>
        <NavLink className="list-group-item active" to="/about">About</NavLink>
        </nav>

        <main>
            <div className="page-header">
                <div className="col-12">
                    <h2 className="fw-normal ms-3">About</h2>
                </div>
            </div>

             <div className="container-fluid">
                <div className="card">
                    <div className="card-body">
                        <h4>Moneta Web</h4>
                        <a target="_blank" rel="noopener noreferrer" href="linktowebsite">url</a>
                    </div>
                </div>
            </div>
        </main>
    </div>
    )
}