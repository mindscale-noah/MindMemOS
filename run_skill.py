import asyncio

from mindmemos_skill import create_service


async def main():
    async with await create_service("config/mindmemos_skill/default.yaml") as service:
        print(service)


if __name__ == "__main__":
    asyncio.run(main())
