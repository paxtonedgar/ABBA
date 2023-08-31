import {useState} from 'react';
import{Collapse, Nav, NavBar, NavbarBrand, NavbarToggler, NavItem} from 'reactstrap';
import {NavLink} form 'react-router-dom';

import {BootstrapThemeToggler, useTheme} from '@moneta/moneta-web-direct';

import logo from '../../assets.logo.svg';

export default function Headers() {
    const [isOpen, setIsOpen] = useState(false);

    const handlNavbarToggle = () => setIsOpen(!isOpen);

    const{isLightTheme}= useTheme();

    return(
        <header>
            <Navbar expand='md' className={isLightTheme ? 'nav-light-bg-light': 'navbar-dark bg-dark'}>
                <navbarBrand tag={NavLink} to "/">
                    <img src={logo} alt="Logo" /><strong>CIU</strong>
                </navbarBrand>
                <NavbarToggler onClick={handlNavbarToggle}>
                    <i className="fas fa-bars" />
                </NavbarToggler>

                <Collapse isOpen={isOpen navbar}>
                    <Nav className = "me-auto navbar">
                        <NavItem>
                            <NavLink className="nav-link" to="/onboard-event">Onboard</NavLink>
                        </NavItem>
                    </Nav>
                    <Nav className = "me-auto navbar">
                        <NavItem>
                            <NavLink className="nav-link" to="/about">About</NavLink>
                        </NavItem>
                    </Nav>
                    <nav className="me-0 navbar-nav">
                        <BootstrapThemeToggler />
                    </nav>
                </Collapse>
            </Navbar>
        </header>
    );
}