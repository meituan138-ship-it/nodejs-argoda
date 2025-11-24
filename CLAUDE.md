# CLAUDE.md - AI Assistant Development Guide

This document provides comprehensive information about the nodejs-argo codebase for AI assistants working with this repository.

## Project Overview

**Project Name:** nodejs-argo
**Version:** 1.0.0
**License:** GPL 3.0
**Language:** JavaScript (Node.js)
**Primary Language:** Chinese (中文)

### Purpose
nodejs-argo is an Argo tunnel deployment tool designed specifically for PaaS platforms and Node.js hosting environments. It provides proxy capabilities supporting multiple protocols (VLESS, VMess, Trojan) and integrates with Nezha monitoring probe functionality.

### Key Features
- Argo tunnel deployment for Node.js environments
- Multiple proxy protocol support
- Nezha probe integration (v0/v1)
- Docker container support
- Temporary and fixed tunnel modes
- Subscription generation
- Auto-keepalive functionality

## Repository Structure

```
nodejs-argoda/
├── index.js              # Main application file (obfuscated/minified)
├── package.json          # Node.js dependencies and scripts
├── Dockerfile           # Docker container configuration
├── README.md            # Chinese documentation
├── LICENSE              # GPL 3.0 license
└── .github/
    └── workflows/
        └── build-docker-image.yml  # Docker build automation
```

## Codebase Characteristics

### index.js
- **Size:** ~112KB single-line minified/obfuscated JavaScript
- **Structure:** Heavily obfuscated code using variable name mangling
- **Framework:** Uses Express.js for HTTP server
- **Cannot be easily modified:** Code is obfuscated and not meant for direct editing
- **Function:** Sets up HTTP server, manages Argo tunnels, handles subscriptions

### Dependencies
```json
{
  "express": "^4.18.2",  // HTTP server framework
  "axios": "latest"      // HTTP client for external requests
}
```

### Runtime Requirements
- Node.js >= 14
- npm/yarn/pnpm for package management
- Docker (optional, for containerized deployment)

## Environment Variables

The application is configured entirely through environment variables:

### Required Variables
None - all variables have defaults for basic operation

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_URL` | - | Subscription upload URL |
| `PROJECT_URL` | https://www.google.com | Project assigned domain |
| `AUTO_ACCESS` | false | Enable auto-keepalive |
| `PORT` | 3000 | HTTP service port |
| `ARGO_PORT` | 8001 | Argo tunnel port |
| `UUID` | 89c13786-25aa-4520-b2e7-12cd60fb5202 | User UUID |
| `NEZHA_SERVER` | - | Nezha panel domain |
| `NEZHA_PORT` | - | Nezha port |
| `NEZHA_KEY` | - | Nezha secret key |
| `ARGO_DOMAIN` | - | Argo fixed tunnel domain |
| `ARGO_AUTH` | - | Argo fixed tunnel authentication |
| `CFIP` | www.visa.com.tw | Node optimization domain/IP |
| `CFPORT` | 443 | Node port |
| `NAME` | Vls | Node name prefix |
| `FILE_PATH` | ./tmp | Working directory |
| `SUB_PATH` | sub | Subscription path |

### Tunnel Modes
- **Temporary Tunnel:** Don't set `ARGO_DOMAIN` and `ARGO_AUTH`
- **Fixed Tunnel:** Set both `ARGO_DOMAIN` and `ARGO_AUTH`

### Nezha TLS
When `NEZHA_PORT` is one of {443, 8443, 2096, 2087, 2083, 2053}, TLS is automatically enabled.

## Development Workflow

### Local Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Or run in production mode
npm start
```

### Docker Development

```bash
# Build Docker image
docker build -t nodejs-argo .

# Run container
docker run -p 3000:3000 \
  -e PORT=3000 \
  -e UUID=your-uuid \
  nodejs-argo
```

### Testing Deployment

Access subscription at:
- Standard port: `https://your-domain.com/sub`
- Custom port: `http://your-domain.com:port/sub`

## CI/CD Pipeline

### GitHub Actions Workflow

**File:** `.github/workflows/build-docker-image.yml`

**Triggers:**
- Push to `main` branch (when Dockerfile, index.js, or package.json changes)
- Pull requests to `main` branch
- Manual workflow dispatch

**Build Process:**
1. Checkout repository
2. Login to GitHub Container Registry (ghcr.io)
3. Set up Docker Buildx
4. Build multi-platform image (linux/amd64, linux/arm64)
5. Push to `ghcr.io/{owner}/mode:9527`
6. Cache layers for faster subsequent builds

### Docker Image Details
- **Registry:** GitHub Container Registry (ghcr.io)
- **Tag:** `mode:9527`
- **Platforms:** linux/amd64, linux/arm64
- **Base Image:** node:alpine3.20

## Git Branch Strategy

### Current Branch
`claude/claude-md-midawx67jhj5sqe8-01Wvrp2WgY3RkKNS8kXwGTmJ`

### Branch Naming Convention
- Claude development branches: `claude/claude-md-{session-id}`
- Main branch: `main`

### Commit Guidelines
1. Use clear, descriptive commit messages
2. Commit messages should be in English for AI assistant work
3. Group related changes together
4. Follow conventional commit format when possible

### Push Requirements
- Always use: `git push -u origin <branch-name>`
- Branch must start with `claude/` and end with matching session ID
- Network failures: retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

## Key Conventions for AI Assistants

### Code Modification Restrictions

**CRITICAL:** The main application code (index.js) is obfuscated and should NOT be modified directly.

**What AI assistants CAN do:**
- Modify configuration files (package.json, Dockerfile)
- Update documentation (README.md, this file)
- Modify GitHub Actions workflows
- Create new scripts or tools around the application
- Update environment variable configurations

**What AI assistants SHOULD NOT do:**
- Attempt to deobfuscate or modify index.js
- Change the core application logic
- Remove or bypass security features

### Documentation Standards
- Keep Chinese documentation in README.md
- Keep English technical documentation in CLAUDE.md
- Update both files when making significant changes
- Maintain consistency between English and Chinese docs

### Docker Considerations
- Base image: node:alpine3.20 (lightweight)
- Required Alpine packages: openssl, curl, gcompat, iproute2, coreutils, bash
- Default working directory: /tmp
- Exposed port: 3000/tcp
- Container runs with: `node index.js`

### Security Considerations
1. **License Compliance:** GPL 3.0 - personal use only, no commercial use
2. **Legal Warning:** Must comply with local laws, no public proxy abuse
3. **Secrets Management:** Never commit sensitive data (UUIDs, keys, tokens)
4. **Environment Variables:** Use for all configuration, not hardcoded values

## Common Tasks

### Updating Dependencies
```bash
npm update
# Test the application still works
npm start
```

### Building New Docker Image
```bash
docker build -t nodejs-argo:test .
docker run -p 3000:3000 nodejs-argo:test
```

### Updating Documentation
1. Modify README.md (Chinese) for user-facing changes
2. Modify CLAUDE.md (English) for technical/AI context
3. Ensure consistency between both documents
4. Commit with clear message describing changes

### Creating Pull Requests
1. Work on feature branch starting with `claude/`
2. Test changes locally first
3. Update relevant documentation
4. Create PR with clear description
5. PR title should describe the change clearly

## Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Change PORT environment variable
export PORT=3001
npm start
```

**Docker Build Fails:**
- Ensure Docker daemon is running
- Check network connectivity for npm install
- Verify Dockerfile syntax

**Application Won't Start:**
- Check Node.js version (must be >= 14)
- Verify all dependencies installed: `npm install`
- Check port availability
- Review environment variable configuration

### Debug Mode
```bash
# Enable Node.js debug output
NODE_DEBUG=* npm start

# Or with specific modules
NODE_DEBUG=http,net npm start
```

## External Resources

- **GitHub Repository:** https://github.com/eooce/nodejs-argo
- **NPM Package:** https://www.npmjs.com/package/nodejs-argo
- **Telegram Group:** https://t.me/eooceu
- **Issues:** https://github.com/eooce/nodejs-argo/issues

## Package.json Scripts

```json
{
  "dev": "node index.js",    // Development mode
  "start": "node index.js"   // Production mode
}
```

Note: Both scripts currently run the same command. The application behavior is controlled by environment variables, not by script differences.

## Platform Deployment Notes

### PaaS Platforms
- Upload index.js and package.json
- Configure environment variables in platform dashboard
- Platform must support Node.js >= 14

### Docker Platforms
- Use provided Dockerfile
- Set environment variables in container configuration
- Map port 3000 to desired external port

### Standalone Servers
- Can run with npm, screen, tmux, PM2, or systemd
- See README.md for detailed setup instructions
- Ensure Node.js runtime is installed

## Project-Specific Context

### Language and Localization
- **Primary Users:** Chinese-speaking community
- **Documentation:** README.md in Chinese
- **Code Comments:** Minimal (obfuscated code)
- **Technical Docs:** English (this file)

### Community and Support
- Active Telegram community for support
- GitHub Issues for bug reports
- Sponsored by VPS.Town and ZMTO

### License Restrictions
- GPL 3.0 with additional restrictions
- **Prohibited:** Commercial use, YouTube/Bilibili/TikTok tutorials
- **Prohibited:** Copying to new repositories for commercial purposes
- **Prohibited:** Public proxy abuse
- Personal use only

## AI Assistant Best Practices

1. **Read Before Writing:** Always read existing files before modifying
2. **Respect Obfuscation:** Don't attempt to modify index.js
3. **Test Changes:** Verify changes work locally when possible
4. **Document Updates:** Update this file when making significant changes
5. **Environment First:** Use environment variables for configuration
6. **Security Aware:** Never commit secrets or sensitive data
7. **License Compliance:** Ensure all work complies with GPL 3.0
8. **Bilingual Support:** Maintain both English and Chinese documentation
9. **Docker Testing:** Test Docker builds after Dockerfile changes
10. **Branch Naming:** Follow the claude/* branch naming convention

## Version History

- **v1.0.0** (Current): Initial release with obfuscated codebase
- Environment-driven configuration
- Docker and PaaS platform support
- Multi-protocol proxy support

---

**Last Updated:** 2025-11-24
**Maintained By:** AI Assistants working with this repository
**For Questions:** Refer to README.md or open a GitHub issue
