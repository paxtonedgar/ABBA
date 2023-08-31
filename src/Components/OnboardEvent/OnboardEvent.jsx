import React, { useState } from 'react';
import './index.scss'; // Import the index.scss file
import { Nav, Navbar, NavItem, NavLink, Collapse } from 'react-bootstrap'; // Import necessary components

function OnboardEvent() {
    const [source, setSource] = useState('');
    const [target, setTarget] = useState('');
    const [open, setOpen] = useState(false);

    const handleSubmit = async () => {
        // ... your submit logic ...
    };

    return (
        <div>
            <Navbar expand="lg">
                <Navbar.Toggle aria-controls="onboard-event-navbar" onClick={() => setOpen(!open)} />
                <Collapse in={open}>
                    <Nav className="mr-auto">
                        <NavItem>
                            <NavLink href="/home">Home</NavLink>
                        </NavItem>
                        {/* Other nav items */}
                    </Nav>
                </Collapse>
            </Navbar>

            <div>
                <label>Source:</label>
                <textarea value={source} onChange={e => setSource(e.target.value)} />
            </div>
            <div>
                <label>Target:</label>
                <textarea value={target} onChange={e => setTarget(e.target.value)} />
            </div>
            <button onClick={handleSubmit}>Submit</button>
        </div>
    );
}

export default OnboardEvent;
