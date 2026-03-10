/**
 * ABBA MCP Server — Cloudflare Worker
 *
 * Thin MCP proxy: registers all ABBA tools and forwards calls
 * to the Python HTTP backend. Deploy this to Cloudflare Workers
 * for a public HTTPS MCP endpoint.
 *
 * Architecture:
 *   Claude/GPT → MCP (Streamable HTTP) → CF Worker → ABBA Python backend
 *
 * The worker fetches the tool list from the backend at startup,
 * registers each as an MCP tool, and proxies calls through.
 */

import { createMcpHandler } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

interface Env {
  ABBA_BACKEND_URL: string;
}

/** Map ABBA param types to Zod schemas */
function paramToZod(paramInfo: Record<string, any>): z.ZodTypeAny {
  const ptype = paramInfo.type || "string";
  let schema: z.ZodTypeAny;

  switch (ptype) {
    case "integer":
      schema = z.number().int();
      break;
    case "number":
      schema = z.number();
      break;
    case "boolean":
      schema = z.boolean();
      break;
    case "object":
      schema = z.record(z.unknown());
      break;
    default:
      schema = z.string();
  }

  if (paramInfo.enum) {
    schema = z.enum(paramInfo.enum);
  }

  if (paramInfo.optional || !paramInfo.required) {
    schema = schema.optional();
  }

  return schema;
}

/** Fetch tool definitions from the Python backend */
async function fetchToolDefs(backendUrl: string): Promise<any[]> {
  const resp = await fetch(`${backendUrl}/tools`);
  if (!resp.ok) throw new Error(`Backend /tools returned ${resp.status}`);
  const data: any = await resp.json();
  return data.tools || [];
}

/** Call a tool on the Python backend */
async function callBackendTool(
  backendUrl: string,
  toolName: string,
  args: Record<string, any>,
): Promise<any> {
  const resp = await fetch(`${backendUrl}/tools/${toolName}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  if (!resp.ok) {
    return { error: `Backend returned ${resp.status}: ${await resp.text()}` };
  }
  return resp.json();
}

function createServer(tools: any[], backendUrl: string): McpServer {
  const server = new McpServer({
    name: "ABBA Sports Analytics",
    version: "2.0.0",
  });

  for (const tool of tools) {
    const params = tool.params || {};
    const zodShape: Record<string, z.ZodTypeAny> = {};

    for (const [paramName, paramInfo] of Object.entries(params)) {
      zodShape[paramName] = paramToZod(paramInfo as Record<string, any>);
    }

    // Register each ABBA tool as an MCP tool
    server.tool(
      tool.name,
      tool.description || "",
      zodShape,
      async (args: Record<string, any>) => {
        const result = await callBackendTool(backendUrl, tool.name, args);
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(result, null, 2),
            },
          ],
        };
      },
    );
  }

  return server;
}

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === "/health") {
      try {
        const resp = await fetch(`${env.ABBA_BACKEND_URL}/health`);
        const data: any = await resp.json();
        return Response.json({
          worker: "ok",
          backend: data,
        });
      } catch (e: any) {
        return Response.json({
          worker: "ok",
          backend: { error: e.message },
        });
      }
    }

    // MCP endpoint — fetch tools from backend and create server
    try {
      const tools = await fetchToolDefs(env.ABBA_BACKEND_URL);
      const server = createServer(tools, env.ABBA_BACKEND_URL);
      const handler = createMcpHandler(server);
      return handler(request, env, ctx);
    } catch (e: any) {
      return Response.json(
        { error: `Failed to initialize: ${e.message}` },
        { status: 502 },
      );
    }
  },
};
