use vulkano::{VulkanLibrary, instance::{Instance, InstanceCreateInfo}, device::{Device, DeviceCreateInfo, QueueCreateInfo}, buffer::{CpuAccessibleBuffer, BufferUsage}, memory::allocator::StandardMemoryAllocator, image::{StorageImage, ImageDimensions, view::ImageView}, format::Format, pipeline::{ComputePipeline, Pipeline}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, CopyImageToBufferInfo}, sync, sync::GpuFuture, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator}};

use vulkano::pipeline::PipelineBindPoint;

use image::{ImageBuffer, Rgba};

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

    let instance = Instance::new(library, InstanceCreateInfo::application_from_cargo_toml()).expect("create instance failed");

    let physical = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices");

    for dev in physical {
        println!("Device {:?} found", dev.properties().device_name);
    }

    let phy_device = instance
        .enumerate_physical_devices()
        .expect("could not enumerate devices")
        .next()
        .expect("device not found");
    
    println!("Device {:?} chosen", phy_device.properties().device_name);

    for family in phy_device.queue_family_properties() {
        println!("Find a queue family with {:?} queues with characteristic {:?}", 
            family.queue_count, family.queue_flags);
    }

    let queue_family_index = phy_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.graphics)
        .expect("couldn't find a graphical queue family") as u32;

    println!("Find queue family {:?} with graphics", queue_family_index);

    let (device, mut queues) = Device::new(
        phy_device, 
        DeviceCreateInfo {
            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    ).expect("fail to create device");

    // println!("Device created on {:?}, with {:?} queues", 
    //     device.physical_device().properties().device_name, queues.count());

    let queue = queues.next().unwrap();

    mod cs {
        vulkano_shaders::shader!{
            ty: "compute",
            path: "shaders/cshader.comp"
        }
    }

    let shader = cs::load(device.clone()).expect("failed to create shader module");

    let shader_entry = shader.entry_point("main").expect("shader entry point not found");
    let shader_layout_req = shader_entry.descriptor_requirements();
    for req in shader_layout_req {
        println!("Shader requirement {:?}", req);
    }

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").expect("shader entry point not found"),
        &(),
        None,
        |_| {}
    ).expect("fail to create pipeline");

    let mem_alloc = StandardMemoryAllocator::new_default(device.clone());

    let image = StorageImage::new(
        &mem_alloc,
        ImageDimensions::Dim2d { 
            width: 1024, 
            height: 1024, 
            array_layers: 1 
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.queue_family_index()),
    ).expect("fail to create image");

    let view = ImageView::new_default(image.clone()).unwrap();

    let buf = CpuAccessibleBuffer::from_iter(
        &mem_alloc,
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        }, 
        false, 
        (0..1024 * 1024 * 4).map(|_| 0u8),
    ).expect("fail to create buffer");
    
    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    let descriptor_set_alloc = StandardDescriptorSetAllocator::new(device.clone());

    let set = PersistentDescriptorSet::new(
        &descriptor_set_alloc,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
    ).expect("fail to create persistent descriptor set");   

    let command_alloc = StandardCommandBufferAllocator::new(device.clone(), StandardCommandBufferAllocatorCreateInfo {
        ..Default::default()
    });

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_alloc, 
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit
    ).expect("fail to create command buffer");

    builder.bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )
        .dispatch([1024 / 8, 1024 / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        )
    ).expect("fail to set up the command buffer");

    let command_buffer = builder.build().expect("fail to build the command buffer");

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

    println!("Everything succeeded!");
}
