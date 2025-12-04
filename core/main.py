from devteam.config import settings, DevTeamConfig
import asyncio

async def main():
    print(settings.model_dump_json())
    config = await DevTeamConfig.from_config_file(settings.config_file)
    print("DevTeam Config:")
    print(config.model_dump_json())
    


if __name__ == "__main__":
    asyncio.run(main())
